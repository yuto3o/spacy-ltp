# -*- coding: utf-8 -*-
import warnings
from typing import Dict, Optional

from ltp import LTP, StnSplit
from spacy import blank, Language
from spacy.tokens import Doc
from spacy.util import registry

DEFAULT_TASKS = ('cws', 'pos', 'dep', 'ner')


def load_pipeline(
        pretrained_model_name_or_path: str,
        *,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Dict = None,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs) -> Language:
    """Create a blank nlp object for a given language code with a ltp
    pipeline as part of the tokenizer. To use the default ltp pipeline with
    the same language code, leave the tokenizer config empty. Otherwise, pass
    in the ltp pipeline settings in config['nlp']['tokenizer'].
    name (str): The language code, e.g. 'en' or 'zh'.

    Parameters
    ----------
    pretrained_model_name_or_path: `str` or `os.PathLike`
        Can be either:
            - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co. Valid model
             ids are [`LTP/tiny`, `LTP/small`, `LTP/base`, `LTP/base1`, `LTP/base1`, `LTP/legacy`], the legacy model
             only support cws, pos and ner, but more fast.
            - You can add `revision` by appending `@` at the end of model_id simply like this:
              `dbmdz/bert-base-german-cased@main` Revision is the specific model version to use. It can be a branch
              name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts
               on huggingface.co, so `revision` can be any identifier allowed by git.
            - A path to a `directory` containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
            - `None` if you are both providing the configuration and state dictionary (resp. with keyword arguments
              `config` and `state_dict`).
    force_download: bool
        Whether to force the (re-)download of the model weights and configuration files, overriding the cached versions
        if they exist.
    resume_download: bool
        Whether to delete incompletely received files. Will attempt to resume the download if such a file exists.
    proxies: Dict[str, str]
        A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
        'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
    use_auth_token:  str or bool
        The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when
        running `transformers-cli login` (stored in `~/.huggingface`).
    cache_dir: str, os.PathLike
        Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
        cache should not be used.
    local_files_only: bool
        Whether to only look at local files (i.e., do not try to download the model).
    kwargs: Dict
        model_kwargs will be passed to the model during initialization

    Returns
    -------
    'Language'
    """
    # Create an empty config skeleton
    config = {'nlp': {'tokenizer': {'kwargs': {}}}}

    # Set the ltp tokenizer
    config['nlp']['tokenizer']['@tokenizers'] = 'spacy_ltp.PipelineAsTokenizer.v1'
    # Set the ltp options
    config['nlp']['tokenizer']['pretrained_model_name_or_path'] = pretrained_model_name_or_path
    config['nlp']['tokenizer']['force_download'] = force_download
    config['nlp']['tokenizer']['resume_download'] = resume_download
    config['nlp']['tokenizer']['proxies'] = proxies
    config['nlp']['tokenizer']['use_auth_token'] = use_auth_token
    config['nlp']['tokenizer']['cache_dir'] = cache_dir
    config['nlp']['tokenizer']['local_files_only'] = local_files_only
    config['nlp']['tokenizer']['kwargs'].update(kwargs)

    # hard code only for chinese
    return blank('zh', config=config)


@registry.tokenizers('spacy_ltp.PipelineAsTokenizer.v1')
def create_tokenizer(
        pretrained_model_name_or_path="LTP/small",
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Dict = None,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
):
    def tokenizer_factory(
            nlp,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **kwargs
    ) -> LtpTokenizer:
        ltp = LTP(pretrained_model_name_or_path=pretrained_model_name_or_path,
                  force_download=force_download,
                  resume_download=resume_download,
                  proxies=proxies,
                  use_auth_token=use_auth_token,
                  cache_dir=cache_dir,
                  local_files_only=local_files_only,
                  **kwargs)

        return LtpTokenizer(
                ltp,
                nlp.vocab,
        )

    return tokenizer_factory


class LtpTokenizer(object):
    """Because we're only running the ltp pipeline once and don't split
    it up into spaCy pipeline components, we'll set all the attributes within
    a custom tokenizer.
    """

    def __init__(self, ltp, vocab):
        """Initialize the tokenizer.

        Parameters
        ----------
        ltp: LTP
            The initialized ltp pipeline.
        vocab: spacy.vocab.Vocab
            The vocabulary to use.

        Returns
        -------
        The custom tokenizer.
        """
        self.ltp = ltp
        self.vocab = vocab
        self.vecs = self._find_embeddings(ltp)

    def __call__(self, text):
        """Convert a ltp instance to a spaCy Doc.

        Parameters
        ----------
        text: str
            The text to process.

        Returns
        -------
        spacy.tokens.Doc, The spaCy Doc object.
        """
        if not text:
            return Doc(self.vocab)
        elif text.isspace():
            return Doc(self.vocab, words=[text], spaces=[False])

        sents = StnSplit().split(text)
        ltp_doc = self.ltp.pipeline(sents, DEFAULT_TASKS)
        ltp_tokens, ltp_heads = self.get_tokens_with_heads(ltp_doc)

        tags = []
        deps = []
        heads = []
        token_texts = [token['text'] for token in ltp_tokens]
        is_aligned = True

        try:
            words, spaces = self.get_words_and_spaces(token_texts, text)
        except ValueError:
            words = token_texts
            spaces = [True] * len(words)
            is_aligned = False
            warnings.warn(
                    'Due to multiword token expansion or an alignment '
                    'issue, the original text has been replaced by space-separated '
                    'expanded tokens.',
                    stacklevel=4,
            )

        offset = 0
        for i, word in enumerate(words):
            if word.isspace() and (
                    i + offset >= len(ltp_tokens) or word != ltp_tokens[i + offset]['text']
            ):
                # insert a space token
                tags.append("SPACE")
                deps.append('')

                # increment any heads left of this position that point beyond
                # this position to the right (already present in heads)
                for j in range(0, len(heads)):
                    if j + heads[j] >= i:
                        heads[j] += 1

                # decrement any heads right of this position that point beyond
                # this position to the left (yet to be added from snlp_heads)
                for j in range(i + offset, len(ltp_heads)):
                    if j + ltp_heads[j] < i + offset:
                        ltp_heads[j] -= 1

                # initial space tokens are attached to the following token,
                # otherwise attach to the preceding token
                if i == 0:
                    heads.append(1)
                else:
                    heads.append(-1)

                offset -= 1
            else:
                token = ltp_tokens[i + offset]
                assert word == token['text']

                tags.append(token['pos'])
                deps.append(token['dep'])
                heads.append(ltp_heads[i + offset])

        doc = Doc(
                self.vocab,
                words=words,
                spaces=spaces,
                tags=tags,
                deps=deps,
                heads=[head + i for i, head in enumerate(heads)],
        )

        ents = set()
        offset = 0
        for sent_id, sent in enumerate(sents):
            for ent_type, ent_text in ltp_doc['ner'][sent_id]:
                for ent_start, ent_end in find_all(sent, ent_text):
                    ents.add((ent_start + offset, ent_end + offset, ent_type))
            offset += len(sent)
        ents = [doc.char_span(ent_start, ent_end, ent_type) for ent_start, ent_end, ent_type in ents]
        if not is_aligned or not all(ents):
            warnings.warn(
                    f"Can't set named entities because of multi-word token "
                    f"expansion or because the character offsets don't map to "
                    f"valid tokens produced by the ltp tokenizer:\n"
                    f"Words: {words}\n"
                    f"Entities: {[(ent_type, ent_text) for ent_type, ent_text in ltp_doc['ner']]}",
                    stacklevel=4,
            )
        else:
            doc.ents = ents

        if self.vecs is not None:
            doc.user_token_hooks['vector'] = self.token_vector
            doc.user_token_hooks['has_vector'] = self.token_has_vector
        return doc

    def get_tokens_with_heads(self, ltp_doc):
        """Flatten the tokens in the ltp Doc and extract the token indices of the sentence start tokens to set
        is_sent_start.

        Parameters
        ----------
        ltp_doc: ltp.Document:
            The processed ltp doc.

        Returns
        -------
        The tokens (words) and heads (deps)
        """
        tokens = []
        heads = []
        offset = 0

        for sent_id, sent_text in enumerate(ltp_doc['cws']):
            for token_id, token in enumerate(sent_text):

                head = ltp_doc['dep'][sent_id]['head'][token_id]
                # Here, we're calculating the absolute token index in the doc,
                # then the *relative* index of the head, -1 for zero-indexed
                # and if the governor is 0 (root), we leave it at 0
                if head:
                    head = head - len(heads) - 1 + offset
                else:
                    head = 0

                heads.append(head)
                tokens.append({'text': token,
                               'pos' : ltp_doc['pos'][sent_id][token_id],
                               'dep' : ltp_doc['dep'][sent_id]['label'][token_id]})
            offset += len(sent_text)
        return tokens, heads

    def get_words_and_spaces(self, words, text):
        if ''.join(''.join(words).split()) != ''.join(text.split()):
            raise ValueError('Unable to align mismatched text and words.')
        text_words = []
        text_spaces = []
        text_pos = 0
        # normalize words to remove all whitespace tokens
        norm_words = [word for word in words if not word.isspace()]
        # align words with text
        for word in norm_words:
            try:
                word_start = text[text_pos:].index(word)
            except ValueError:
                raise ValueError('Unable to align mismatched text and words.')
            if word_start > 0:
                text_words.append(text[text_pos: text_pos + word_start])
                text_spaces.append(False)
                text_pos += word_start
            text_words.append(word)
            text_spaces.append(False)
            text_pos += len(word)
            if text_pos < len(text) and text[text_pos] == ' ':
                text_spaces[-1] = True
                text_pos += 1
        if text_pos < len(text):
            text_words.append(text[text_pos:])
            text_spaces.append(False)
        return (text_words, text_spaces)

    def token_vector(self, token):
        """Get ltp's pretrained word embedding for given token.

        Parameters
        ----------
        token: Token
            The token whose embedding will be returned

        Returns
        -------
        np.ndarray, the embedding/vector.
            token.vector.size > 0 if ltp pipeline contains a processor with
            embeddings, else token.vector.size == 0. A 0-vector (origin) will be returned
            when the token doesn't exist in ltp's pretrained embeddings.
        """
        unit_id = self.ltp.tokenizer.convert_tokens_to_ids(token)
        return self.vecs[unit_id]

    def token_has_vector(self, token):
        """Check if the token exists as a unit in ltp's pretrained embeddings.
        """
        return self.ltp.tokenizer.convert_tokens_to_ids(token) != self.ltp.tokenizer.unk_token_id

    @staticmethod
    def _find_embeddings(ltp):
        """Find pretrained word embeddings in any of a LTP's processors.
        """
        embs = None

        if hasattr(ltp.model.backbone, 'embeddings') and hasattr(ltp.model.backbone.embeddings, 'word_embeddings'):
            embs = ltp.model.backbone.embeddings.word_embeddings.weight.detach().cpu().numpy()

        return embs

    # dummy serialization methods
    def to_bytes(self, **kwargs):
        return b''

    def from_bytes(self, _bytes_data, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self


def find_all(s, sub):
    """Find all start index of subtext in the text

    Parameters
    ----------
        s: str
        sub: str

    Returns
    -------
        List[int]
    """
    if len(sub) == 0:
        return []

    p = s.find(sub)
    q = []
    while p != -1:
        q.append((p, p + len(sub)))
        p = s.find(sub, p + len(sub))

    return q


__all__ = [
        'load_pipeline',
]
