# inspired by https://www.oreilly.com/learning/caption-this-with-tensorflow
import tensorflow as tf
import glob
import os
import argparse
from recognize_common import parse_label
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import itertools

tf.logging.set_verbosity(tf.logging.INFO)

mark_start = 'ssss '
mark_end = ' eeee'

def mark_captions(captions_list, start=True):
    captions_marked = [(mark_start if start else "") + caption + mark_end
                        for caption in captions_list]

    return captions_marked

class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""

    def __init__(self, texts, num_words=None):
        """
        :param texts: List of strings with the data-set.
        :param num_words: Max number of words to use.
        """

        Tokenizer.__init__(self, num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]

        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text


def get_random_caption_tokens(idx, tokens):
    """
    Given a list of indices for images in the training-set,
    select a token-sequence for a random caption,
    and return a list of all these token-sequences.
    """
    return np.asarray([tokens[i] for i in idx])


def batch_generator(batch_size, train_embeddings, tokens):
    """
    Generator function for creating random batches of training-data.

    Note that it selects the data completely randomly for each
    batch, corresponding to sampling of the training-set with
    replacement. This means it is possible to sample the same
    data multiple times within a single epoch - and it is also
    possible that some data is not sampled at all within an epoch.
    However, all the data should be unique within a single batch.
    """

    # Infinite loop.
    while True:
        # Get a list of random indices for images in the training-set.
        idx = np.random.randint(len(train_embeddings),
                                size=batch_size)

        transfer_values = np.asarray([train_embeddings[i] for i in idx])

        # For each of the randomly chosen images there are
        # at least 5 captions describing the contents of the image.
        # Select one of those captions at random and get the
        # associated sequence of integer-tokens.
        chosen_tokens = get_random_caption_tokens(idx, tokens)

        # Count the number of tokens in all these token-sequences.
        max_tokens = np.max([len(t) for t in chosen_tokens])

        # Pad all the other token-sequences with zeros
        # so they all have the same length and can be
        # input to the neural network as a numpy array.
        tokens_padded = pad_sequences(chosen_tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')

        # Further prepare the token-sequences.
        # The decoder-part of the neural network
        # will try to map the token-sequences to
        # themselves shifted one time-step.
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        # Dict for the input-data. Because we have
        # several inputs, we use a named dict to
        # ensure that the data is assigned correctly.
        x_data = \
        {
            'decoder_input': decoder_input_data,
            'transfer_values_input': transfer_values
        }

        # Dict for the output-data.
        y_data = \
        {
            'decoder_output': decoder_output_data
        }

        yield (x_data, y_data)


def main(unused_argv):
    argp = argparse.ArgumentParser()
#    argp.add_argument("--max_steps", help="Max steps to train the model",
#                      default=1000, type=int)
#    argp.add_argument("--nocrop", help="If set images will not be croped",
#                      action="store_true")
    argp.add_argument("input", help="Directory with images and labels")
    args = argp.parse_args()

    embedding_paths = glob.glob(os.path.join(args.input, "embedding*npy"))
    embeddings = list(map(lambda x: np.load(x), embedding_paths))
    label_paths = list(map(lambda path:
                           path.replace("embedding", "embedding_label")
                           .replace(".npy", ""),
                           embedding_paths))
    def read_label(path):
        with open(path) as f:
            return f.read().strip()
    labels = list(map(read_label, label_paths))

    train_size = int(len(embeddings) * 0.7)
    train_embeddings = embeddings[:train_size]
    test_embeddings = embeddings[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    num_words = 9
    captions_train_marked = mark_captions(train_labels)
    tokenizer = TokenizerWrap(texts=captions_train_marked,
                              num_words=num_words)
    tokens_train = tokenizer.texts_to_sequences(captions_train_marked)

    generator = batch_generator(100, train_embeddings, tokens_train)
    batch = next(generator)

    state_size = 512
    embedding_size = 512
    transfer_values_input = Input(shape=(512,),
            name='transfer_values_input')
    decoder_input = Input(shape=(None, ), name='decoder_input')
    decoder_embedding = Embedding(input_dim=num_words,
            output_dim=embedding_size,
            name='decoder_embedding')
    decoder_gru1 = GRU(state_size, name='decoder_gru1',
                       return_sequences=True)
    decoder_gru2 = GRU(state_size, name='decoder_gru2',
                       return_sequences=True)
    decoder_gru3 = GRU(state_size, name='decoder_gru3',
                       return_sequences=True)
    decoder_dense = Dense(num_words,
                          activation='linear',
                          name='decoder_output')

    def connect_decoder(transfer_values):
        # Map the transfer-values so the dimensionality matches
        # the internal state of the GRU layers. This means
        # we can use the mapped transfer-values as the initial state
        # of the GRU layers.
#        initial_state = decoder_transfer_map(transfer_values)

        # Start the decoder-network with its input-layer.
        net = decoder_input

        # Connect the embedding-layer.
        net = decoder_embedding(net)

        # Connect all the GRU layers.
        net = decoder_gru1(net, initial_state=transfer_values)
        net = decoder_gru2(net, initial_state=transfer_values)
        net = decoder_gru3(net, initial_state=transfer_values)

        # Connect the final dense layer that converts to
        # one-hot encoded arrays.
        decoder_output = decoder_dense(net)

        return decoder_output

    decoder_output = connect_decoder(transfer_values=transfer_values_input)

    decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                          outputs=[decoder_output])

    def sparse_cross_entropy(y_true, y_pred):
        """
        Calculate the cross-entropy loss between y_true and y_pred.

        y_true is a 2-rank tensor with the desired output.
        The shape is [batch_size, sequence_length] and it
        contains sequences of integer-tokens.

        y_pred is the decoder's output which is a 3-rank tensor
        with shape [batch_size, sequence_length, num_words]
        so that for each sequence in the batch there is a one-hot
        encoded array of length num_words.
        """

        # Calculate the loss. This outputs a
        # 2-rank tensor of shape [batch_size, sequence_length]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                              logits=y_pred)

        # Keras may reduce this across the first axis (the batch)
        # but the semantics are unclear, so to be sure we use
        # the loss across the entire 2-rank tensor, we reduce it
        # to a single scalar with the mean function.
        loss_mean = tf.reduce_mean(loss)
        return loss_mean


    optimizer = RMSprop(lr=1e-3)
    decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
    decoder_model.compile(optimizer=optimizer,
            loss=sparse_cross_entropy,
            target_tensors=[decoder_target])

    path_checkpoint = '22_checkpoint.keras'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                          verbose=1,
                                          save_weights_only=True)

    callback_tensorboard = TensorBoard(log_dir='./22_logs/',
                                       histogram_freq=0,
                                       write_graph=False)
    callbacks = [callback_checkpoint, callback_tensorboard]

    try:
        decoder_model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

#    decoder_model.fit_generator(generator=generator,
#            steps_per_epoch=1000,
#            epochs=100,
#            callbacks=callbacks)

    token_start = tokenizer.word_index[mark_start.strip()]
    token_end = tokenizer.word_index[mark_end.strip()]
    correct = 0
    def generate_caption(embedding, real_label, max_tokens=30):
        transfer_values = np.asarray([embedding])

        # Pre-allocate the 2-dim array used as input to the decoder.
        # This holds just a single sequence of integer-tokens,
        # but the decoder-model expects a batch of sequences.
        shape = (1, max_tokens)
        decoder_input_data = np.zeros(shape=shape, dtype=np.int)

        # The first input-token is the special start-token for 'ssss '.
        token_int = token_start

        # Initialize an empty output-text.
        output_text = ''

        # Initialize the number of tokens we have processed.
        count_tokens = 0

        # While we haven't sampled the special end-token for ' eeee'
        # and we haven't processed the max number of tokens.
        while token_int != token_end and count_tokens < max_tokens:
            # Update the input-sequence to the decoder
            # with the last token that was sampled.
            # In the first iteration this will set the
            # first element to the start-token.
            decoder_input_data[0, count_tokens] = token_int

            # Wrap the input-data in a dict for clarity and safety,
            # so we are sure we input the data in the right order.
            x_data = \
            {
                'transfer_values_input': transfer_values,
                'decoder_input': decoder_input_data
            }

            # Note that we input the entire sequence of tokens
            # to the decoder. This wastes a lot of computation
            # because we are only interested in the last input
            # and output. We could modify the code to return
            # the GRU-states when calling predict() and then
            # feeding these GRU-states as well the next time
            # we call predict(), but it would make the code
            # much more complicated.

            # Input this data to the decoder and get the predicted output.
            decoder_output = decoder_model.predict(x_data)

            # Get the last predicted token as a one-hot encoded array.
            # Note that this is not limited by softmax, but we just
            # need the index of the largest element so it doesn't matter.
            token_onehot = decoder_output[0, count_tokens, :]

            # Convert to an integer-token.
            token_int = np.argmax(token_onehot)

            # Lookup the word corresponding to this integer-token.
            sampled_word = tokenizer.token_to_word(token_int)

            # Append the word to the output-text.
            output_text += " " + sampled_word

            # Increment the token-counter.
            count_tokens += 1

        # This is the sequence of tokens output by the decoder.
        output_tokens = decoder_input_data[0]

        # Print the predicted caption.
        print("Predicted caption:")
        output_text = " ".join(output_text.strip().split()[:-1])
        xx = real_label.split("\t")
        perms = list(itertools.permutations(xx))
        perms = [" ".join(x) for x in perms]
        print(output_text, real_label, output_text in perms)
        nonlocal correct
        correct += 1 if output_text in perms else 0

    captions_test_marked = mark_captions(test_labels, False)
    tokens_test = tokenizer.texts_to_sequences(captions_test_marked)

    test_embeddings = test_embeddings[:1000]
    test_labels = test_labels[:1000]
    for embedding, real_label in list(zip(test_embeddings, test_labels))[::10]:
        generate_caption(embedding, real_label)
    print(float(correct) / len(test_embeddings) * 10)


if __name__ == "__main__":
    tf.app.run()
