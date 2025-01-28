import torch
from torch import Tensor
from typing import Callable, List, Optional, Tuple

from torch.nn.utils.rnn import pad_sequence

from cleaner import TextProcessor
from phonemizers.espeak_wrapper import ESpeak


char_to_id = {'': 0, '[PAD]': 69, '[UNK]': 68, 'aɪ': 2, 'aɪə': 3, 'aɪɚ': 4, 'aʊ': 5, 'b': 6, 'd': 7, 'dt': 8, 'dʒ': 9, 'eɪ': 10, 'f': 11, 'h': 12, 'h#': 13, 'i': 14, 'iə': 15, 'iː': 16, 'j': 17, 'k': 18, 'l': 19, 'm': 20, 'n': 21, 'n̩': 22, 'oʊ': 23, 'p': 24, 'r': 25, 's': 26, 't': 27, 'tʃ': 28, 'uː': 29, 'v': 30, 'w': 31, 'x': 32, 'z': 33, '|': 1, 'æ': 34, 'ç': 35, 'ð': 36, 'ŋ': 37, 'ɐ': 38, 'ɑː': 39, 'ɑːɹ': 40, 'ɑ̃': 41, 'ɔ': 42, 'ɔɪ': 43, 'ɔː': 44, 'ɔːɹ': 45, 'ɔ̃': 46, 'ə': 47, 'əl': 48, 'ɚ': 49, 'ɛ': 50, 'ɛɹ': 51, 'ɜː': 52, 'ɡ': 53, 'ɡʲ': 54, 'ɪ': 55, 'ɪɹ': 56, 'ɬ': 57, 'ɹ': 58, 'ɾ': 59, 'ʃ': 60, 'ʊ': 61, 'ʊɹ': 62, 'ʌ': 63, 'ʒ': 64, 'ʔ': 65, 'θ': 66, 'ᵻ': 67}

# default map

LANGUAGE_MAP = {
    'en-us': 'en',
    'fr-fr': 'es',
    'hi': 'hi'
}

# functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# main class

class Tokenizer:
    def __init__(
        self,
        char_to_id = char_to_id,
        text_cleaner: Optional[Callable] = None,
        phonemizer: Optional[Callable] = None,
        default_lang = "en-us",
        add_blank: bool = False,
        use_eos_bos = False,
        pad_id = -1
    ):
        self.text_cleaner = default(text_cleaner, TextProcessor().phoneme_cleaners)
        self.add_blank = add_blank
        self.use_eos_bos = use_eos_bos
        self.pad_id = pad_id

        self.vocab = list(char_to_id.keys())
        self.vocab_size = len(char_to_id)

        self.char_to_id = char_to_id
        self.id_to_char = {v: k for k, v in char_to_id.items()}

        self.phonemizer = phonemizer
        if not exists(self.phonemizer):
            self.phonemizer = ESpeak(language = default_lang)

        # self.language = self.phonemizer.language
        self.language = default_lang
        self.not_found_characters = []

    @property
    def espeak_language(self):
        return LANGUAGE_MAP.get(self.language, None)

    def encode(self, text: str) -> List[int]:
        """Encodes a string of text as a sequence of IDs."""
        token_ids = []
        for char in text:
            try:
                idx = self.char_to_id[char]
                token_ids.append(idx)
            except KeyError:
                idx = self.char_to_id['[UNK]']
                token_ids.append(idx)
                # # discard but store not found characters
                # if char not in self.not_found_characters:
                #     self.not_found_characters.append(char)
                #     print(text)
                #     print(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a sequence of IDs to a string of text."""
        text = ""
        for token_id in token_ids:
            text += self.id_to_char[token_id]
        return text

    def text_to_ids(
        self,
        text: str,
        language: str = None
    ) -> Tuple[List[int], str, str]:
        """Converts a string of text to a sequence of token IDs.

        Args:
            text(str):
                The text to convert to token IDs.

            language(str):
                The language code of the text. Defaults to None.

        TODO:
            - Add support for language-specific processing.

        1. Text normalizatin
        2. Phonemization (if use_phonemes is True)
        3. Add blank char between characters
        4. Add BOS and EOS characters
        5. Text to token IDs
        """
        try:
            language = default(language, self.espeak_language)

            cleaned_text = None
            if self.text_cleaner is not None:
                # text = self.text_cleaner(text, language=language)
                text = self.text_cleaner(text)
                cleaned_text = text
            # phonemized = self.phonemizer.phonemize(text, separator="", language=language)
            print(text)
            phonemized = self.phonemizer.phonemize(text, separator=" h# ").split()
            print(phonemized)
            if self.add_blank:
                phonemized = self.intersperse_blank_char(phonemized, True)
            if self.use_eos_bos:
                phonemized = self.pad_with_bos_eos(phonemized)

            return self.encode(phonemized), cleaned_text, phonemized
        except Exception as e:
            print(f"Error processing text: {text}")
            print(f"Exception: {e}")
            raise

    def texts_to_tensor_ids(self, texts: List[str], language: str = None) -> Tensor:
        all_ids = []

        for text in texts:
            ids, *_ = self.text_to_ids(text, language = language)
            all_ids.append(torch.tensor(ids))

        return pad_sequence(all_ids, batch_first = True, padding_value = self.pad_id)



    def phoneme_to_tensor_ids(self, texts: List[str], padding_value) -> Tensor:
        all_ids = []

        for text in texts:
            ids = self.encode(text)
            all_ids.append(torch.tensor(ids))

        return pad_sequence(all_ids, batch_first = True, padding_value = padding_value)


    def texts_to_tensor_ids_with_phonemes(self, texts: List[str], language: str = None) -> Tensor:
        all_ids = []
        all_phonemes = []
        for text in texts:
            ids, _, phonemized  = self.text_to_ids(text, language = language)
            all_ids.append(torch.tensor(ids))
            all_phonemes.append(phonemized)

        return pad_sequence(all_ids, batch_first = True, padding_value = self.pad_id), all_phonemes

    def ids_to_text(self, id_sequence: List[int]) -> str:
        """Converts a sequence of token IDs to a string of text."""
        return self.decode(id_sequence)

    def pad_with_bos_eos(self, char_sequence: List[str]):
        """Pads a sequence with the special BOS and EOS characters."""
        return [self.characters.bos] + list(char_sequence) + [self.characters.eos]

    def intersperse_blank_char(self, char_sequence: List[str], use_blank_char: bool = False):
        """Intersperses the blank character between characters in a sequence.

        Use the ```blank``` character if defined else use the ```pad``` character.
        """
        char_to_use = self.characters.blank if use_blank_char else self.characters.pad
        result = [char_to_use] * (len(char_sequence) * 2 + 1)
        result[1::2] = char_sequence
        return result
