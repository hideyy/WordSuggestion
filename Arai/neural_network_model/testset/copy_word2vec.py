






class Word2Vec(utils.SaveLoad):
    """
    Class for training, using and evaluating neural networks described in https://code.google.com/p/word2vec/
    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in a format
    compatible with the original word2vec implementation via `wv.save_word2vec_format()` and `KeyedVectors.load_word2vec_format()`.
    """

    def __init__(
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH):

        self.initialize_word_vectors()
        self.sg = int(sg)
        self.cum_table = None  # for negative sampling
        self.vector_size = int(size)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        #学習率
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.random = random.RandomState(seed)
        self.min_count = min_count
        self.sample = sample
        self.workers = int(workers)
        self.min_alpha = float(min_alpha)
        #negative samplingの個数
        self.negative = negative
        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.model_trimmed_post_training = False
        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(sentences, trim_rule=trim_rule)
            self.train(sentences)

    def initialize_word_vectors(self):
        #word
        self.wv = KeyedVectors()
        #category
        self.cv = KeyedVectors()

    #same as word2vec
    def make_cum_table(self, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.
        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.
        Called internally from 'build_vocab()'.
        """
        vocab_size = len(self.wv.index2word)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in xrange(vocab_size):
            train_words_pow += self.wv.vocab[self.wv.index2word[word_index]].count**power
        cumulative = 0.0
        for word_index in xrange(vocab_size):
            cumulative += self.wv.vocab[self.wv.index2word[word_index]].count**power
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.
        """
        self.scan_vocab(sentences, progress_per=progress_per, trim_rule=trim_rule)  # initial survey
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)  # trim by min_count & precalculate downsampling
        self.finalize_vocab(update=update)  # build tables & arrays

    #article = ([categories], sentence)
    def scan_vocab(self, articles, progress_per=10000, trim_rule=None):
        """Do an initial scan of all words appearing in articles."""
        logger.info("collecting all categories, words and their counts")
        article_no = -1
        total_words = 0
        total_categories = 0
        word_vocab = defaultdict(int)
        category_vocab = defaultdict(int)
        checked_string_types = 0
        for article_no, article in enumerate(articles):
            categories = article[0] #list
            sentence = article[1] #string
            if not checked_string_types:
                if isinstance(sentence, string_types):
                    logger.warn("Each 'sentences' item should be a list of words (usually unicode strings)."
                                "First item here is instead plain %s.", type(sentence))
                checked_string_types += 1
            if sentence_no % progress_per == 0:
                logger.info("PROGRESS: at article #%i, processed %i categories, keeping %i category types, %i words, keeping %i word types",
                            article_no, sum(itervalues(category_vocab)) + total_categories, len(category_vocab), sum(itervalues(word_vocab)) + total_words, len(word_vocab))

            for category in categories:
                category_vocab[category] += 1
            for word in sentence:
                word_vocab[word] += 1

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                total_words += utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        total_categories += sum(itervalues(category_vocab))
        total_words += sum(itervalues(word_vocab))

        logger.info("collected %i word types from a corpus of %i raw words and %i articles",
                    len(word_vocab), total_words, article_no + 1)
        self.corpus_count = article_no + 1
        self.raw_category_vocab = category_vocab
        self.raw_word_vocab = word_vocab

    def scale_vocab(self, min_count=None):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).
        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.
        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.
        """
        min_count = min_count or self.min_count
        drop_ctotal = drop_cunique = 0
        drop_wtotal = drop_wunique = 0

        logger.info("Loading a fresh vocabulary")
        retain_ctotal, retain_categories = 0, []
        retain_wtotal, retain_words = 0, []
        # Discard words less-frequent than min_count
        if not dry_run:
            self.cv.index2word = []
            self.wv.index2word = []
            # make stored settings match these applied settings
            self.min_count = min_count
            self.sample = sample

            self.cv.vocab = {}
            self.wv.vocab = {}

        for category, v in iteritems(self.raw_category_vocab):
            if keep_vocab_item(category, v, min_count, trim_rule=trim_rule):
                retain_categories.append(category)
                retain_ctotal += v
                if not dry_run:
                    self.cv.vocab[category] = Vocab(count=v, index=len(self.cv.index2word))
                    self.cv.index2word.append(category)
            else:
                drop_cunique += 1
                drop_ctotal += v

        for word, v in iteritems(self.raw_word_vocab):
            if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                retain_words.append(word)
                retain_wtotal += v
                if not dry_run:
                    self.wv.vocab[word] = Vocab(count=v, index=len(self.wv.index2word))
                    self.wv.index2word.append(word)
            else:
                drop_wunique += 1
                drop_wtotal += v

        original_cunique_total = len(retain_categories) + drop_cunique
        original_wunique_total = len(retain_words) + drop_wunique

        retain_cunique_pct = len(retain_words) * 100 / max(original_cunique_total, 1)
        retain_wunique_pct = len(retain_words) * 100 / max(original_wunique_total, 1)

        logger.info("min_count=%d retains %i unique categories (%i%% of original %i, drops %i)",
                    min_count, len(retain_categories), retain_cunique_pct, original_cunique_total, drop_cunique)
        logger.info("min_count=%d retains %i unique words (%i%% of original %i, drops %i)",
                    min_count, len(retain_words), retain_wunique_pct, original_wunique_total, drop_wunique)

        original_ctotal = retain_ctotal + drop_ctotal
        original_wtotal = retain_wtotal + drop_wtotal
        retain_cpct = retain_ctotal * 100 / max(original_ctotal, 1)
        retain_wpct = retain_wtotal * 100 / max(original_wtotal, 1)
        logger.info("min_count=%d leaves %i word corpus (%i%% of original %i, drops %i)",
                    min_count, retain_ctotal, retain_cpct, original_ctotal, drop_ctotal)
        logger.info("min_count=%d leaves %i word corpus (%i%% of original %i, drops %i)",
                    min_count, retain_wtotal, retain_wpct, original_wtotal, drop_wtotal)


        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.wv.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
                    downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

        # return from each step: words-affected, resulting-corpus-size
        report_values = {'drop_unique': drop_unique, 'retain_total': retain_total,
                         'downsample_unique': downsample_unique, 'downsample_total': int(downsample_total)}

        # print extra memory estimates
        report_values['memory'] = self.estimate_memory(vocab_size=len(retain_words))

        return report_values

    def finalize_vocab(self, update=False):
        """Build tables and model weights based on final vocabulary settings."""
        if not self.wv.index2word:
            self.scale_vocab()
        if self.sorted_vocab and not update:
            self.sort_vocab()
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.wv.vocab)
            self.wv.index2word.append(word)
            self.wv.vocab[word] = v
        # set initial input/projection and hidden weights
        if not update:
            self.reset_weights()
        else:
            self.update_weights()

    def sort_vocab(self):
        """Sort the vocabulary so the most frequent words have the lowest indexes."""
        if len(self.wv.syn0):
            raise RuntimeError("must sort before initializing vectors/weights")
        self.wv.index2word.sort(key=lambda word: self.wv.vocab[word].count, reverse=True)
        for i, word in enumerate(self.wv.index2word):
            self.wv.vocab[word].index = i
