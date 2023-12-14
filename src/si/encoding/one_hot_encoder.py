import numpy as np

class OneHotEncoder:
    """
    Implements a one-hot encoder.
    Representation technique that converts categorical variables into a form that could be provided to ML algorithms to do a better job in prediction.
    """

    def __init__(self, padder: str, max_length: int = None):
        """
        Parameters
        ----------
        padder:str
            character to use for padding
        max_length:int
            maximum length of sequences

        Attributes
        ----------
        alphabet:set
            set of characters in the dataset
        char_to_index:dict
            maps characters to indices
        index_to_char:dict
            maps indices to characters
        """
        self.padder = padder
        self.max_length = max_length

        self.alphabet = set()
        self.char_to_index = {}
        self.index_to_char = {}

    def fit(self, data: list[str]) -> "OneHotEncoder":
        """
        Fit the encoder to the dataset.

        Parameters
        ----------
        X: numpy.ndarray
            The dataset to fit
        """
        if self.max_length is None:
            lengths=[len(sequence) for sequence in data]
            self.max_length = np.max(lengths)

        seq = "".join(data)
        self.alphabet = set(seq)
        indexes = np.arange(1, len(self.alphabet) + 1)  # create an array of indexes from 1 to the length of the alphabet
        self.char_to_index = dict(zip(self.alphabet,indexes))  # create a dictionary that maps each character in the alphabet to its corresponding index
        self.index_to_char = dict(zip(indexes, self.alphabet))

        if self.padder not in self.alphabet:
            self.alphabet.add(self.padder)
            self.char_to_index[self.padder] = len(self.alphabet) + 1
            self.index_to_char[len(self.alphabet) + 1] = self.padder

        for i, char in enumerate(self.alphabet):
            self.char_to_index[char] = i
            self.index_to_char[i] = char

        return self

    def transform(self, data: list[str]) -> np.ndarray:
        """
        Transform the dataset.

        Parameters
        ----------
        X: numpy.ndarray
            The dataset to transform
        Returns
        -------
        numpy.ndarray
            The transformed dataset
        """
        transformed = []
        for sequence in data:
            sequence = list(sequence) #convert to list of characters
            sequence = sequence[:self.max_length] #trimming to max length
            sequence = sequence + [self.padder] * (self.max_length - len(sequence)) #padding so sequences are all the same length

            one_hot_sequence = []
            for char in sequence:
                one_hot_char = np.zeros(len(self.alphabet)) #create a one-hot vector
                one_hot_char[self.char_to_index[char]] = 1  #set the index of the character to 1
                one_hot_sequence.append(one_hot_char)

            transformed.append(one_hot_sequence)

        return np.array(transformed)

    def inverse_transform(self, data: np.ndarray) -> list[str]:
        """
        Inverse the encoding transform on the dataset.

        Parameters
        ----------
        X: numpy.ndarray
            The dataset to inverse transform
        Returns
        -------
        list
            The inverse transformed dataset
        """
        sequences = []
        for one_hot_sequence in data:
            seq = []
            for one_hot_char in one_hot_sequence:
                index = np.argmax(one_hot_char)
                seq.append(self.index_to_char[index])
            sequences.append("".join(seq))

        decoded_sequences = []
        for seq in sequences:
            for i in range(0, len(seq), self.max_length): #analyses chunks of seq in max_length size
                string = seq[i:i + self.max_length]
                trimmed = string.rstrip(self.padder) #removes padding
                decoded_sequences.append(trimmed)

        return decoded_sequences

    def fit_transform(self, data: list[str]) -> np.ndarray:
        """
        Fit the encoder to the dataset and transform it.

        Parameters
        ----------
        X: numpy.ndarray
            The dataset to fit and transform
        Returns
        -------
        numpy.ndarray
            The transformed dataset
        """
        self.fit(data)
        return self.transform(data)


if __name__ == '__main__':
    from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

    encoder = OneHotEncoder(padder="!", max_length=9)
    data = ["abcd", "aabdee"]
    encoded_data = encoder.fit_transform(data)
    print("One-Hot Encoding:")
    print(np.array(encoded_data))
    decoded_data = encoder.inverse_transform(encoded_data)

    print("\nReverting encoded Data:")
    print(decoded_data)

    sklearn_encoder = SklearnOneHotEncoder(sparse=False, handle_unknown='ignore')
    sklearn_data = np.array(data).reshape(-1, 1)
    sklearn_encoded_data = sklearn_encoder.fit_transform(sklearn_data)

    # print("\nScikit-learn One-Hot Encoding:")
    # print(sklearn_encoded_data)

    sklearn_decoded_data = sklearn_encoder.inverse_transform(sklearn_encoded_data)
    print("\nScikit-learn Decoded Data:")
    print(sklearn_decoded_data)