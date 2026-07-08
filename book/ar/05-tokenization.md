# الفصل 5 — التجزئة إلى رموز

*الجزء الثاني، الفصل 1 من 4. بنى الجزء الأول آلة تعلُّم كاملة. والجزء الثاني
يوجِّهها نحو اللغة. المشكلة الأولى: الشبكات العصبية تتغذّى على الأرقام، ولدينا
نصّ.*

## ما الذي يفعله نموذج اللغة حقًّا

انزع الهالة الغامضة، فإذا نموذج اللغة مُصنِّف (classifier) من الفصل 3: بمعلومية
النصّ حتى الآن، **تنبّأ بالقطعة التالية من النصّ**.

```
input:  "To be, or not to b"      ->  model  ->  "e"  (probably)
```

تنبّؤ واحد لا يُثير الإعجاب. الحيلة في *التكرار*: أضِف القطعة المُتنبَّأ بها، وتنبّأ
مرّة أخرى، ثمّ أخرى — فيكتب النموذج (الفصل 8). كلّ قدرة في النماذج على غرار GPT
مُدرَّبة عبر هذا الهدف الواحد: التحسّن في تخمين ما يأتي تاليًا.

لكنّ المُصنِّف يحتاج إلى مجموعة ثابتة من الأصناف (classes) يختار من بينها، وإلى
مُدخلات عددية. لذا، قبل أيّ نمذجة، علينا أن نُقرّر: ما *قِطَع* النصّ؟ يُسمّى هذا
القرار **التجزئة إلى رموز** (tokenization)، وهو يُحدِّد مُفردات النموذج — أبجدية
كلّ ما سيقرؤه أو يقوله يومًا:

```
   text ──encode──►  [18, 47, 32, 1]  ──model──►  [32]  ──decode──►  text
              (token ids: plain integers)
```

يعيش المُجزّئان أدناه في
[`babytorch/text/tokenizers.py`](../../babytorch/text/tokenizers.py)،
وكلاهما خريطة ثنائية الاتّجاه بين السلاسل النصّية والمُعرِّفات الصحيحة.

## أبسط إجابة: رمز واحد لكلّ محرف

يمسح `CharTokenizer` مُدوَّنةً، ويجمع كلّ محرف مُتمايز، ويُرقّمها أبجديًّا:

```python
>>> from babytorch.text import CharTokenizer
>>> tok = CharTokenizer("hello world")
>>> tok.chars                      # the whole vocabulary
[' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
>>> tok.vocab_size
8
>>> tok.encode("low")
[4, 5, 7]
>>> tok.decode([4, 5, 7])
'low'
```

التجزئة على مستوى المحرف صادقة وضئيلة — لا تحتاج Tiny Shakespeare إلّا إلى نحو
65 رمزًا — وهي ما يستخدمه BabyGPT، لأنّها تجعل كلّ جزء من خطّ المعالجة شفّافًا:
لا شيء مخفيّ داخل الرموز. أمّا الثمن: تطول المُتتاليات (رمز واحد لكلّ حرف)، وعلى
النموذج أن يُنفِق طاقته في تعلّم *التهجئة* قبل أن يتعلّم *الكتابة*.

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/text/tokenizers.py</code> (المُجزّئ كلّه، عمليًّا)</summary>

```python
    def fit(self, text):
        """Build the vocabulary from all characters seen in ``text``."""
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}   # string -> int
        self.itos = {i: ch for i, ch in enumerate(self.chars)}   # int -> string
        return self

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        """Text -> list of integer ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """List of ids -> text."""
        return ''.join(self.itos[int(i)] for i in ids)
```

</details>

أمّا النقيض الآخر، رمز واحد لكلّ **كلمة**، فله المشكلة المعكوسة: مُفردات من مئات
الآلاف، وصفّ تضمين (embedding) لكلٍّ منها، وعمًى تامّ عن أيّ كلمة لم تُرَ في
التدريب (`"untokenizable"` →
`???`).

## إجابة GPT: ترميز أزواج البايتات (Byte Pair Encoding)

تجد BPE الحلّ الوسط تلقائيًّا: **دَع المُدوَّنة نفسها تُقرّر ما القِطَع.**
فالمُتتاليات المُتكرّرة تستحقّ رمزها الخاصّ؛ والنادرة يمكن أن تبقى مُقسَّمة إلى
أجزاء. وخوارزمية التدريب أربعة أسطر بسيطة:

1. ابدأ بالمُفردات = المحارف المُفردة.
2. عُدَّ كلّ زوج مُتجاور من الرموز في المُدوَّنة.
3. ادمج الزوج الأكثر تكرارًا في رمز جديد واحد.
4. كرِّر حتى تبلغ المُفردات الحجم المُستهدَف.

راقبها تعمل على مُدوَّنة من ستّ كلمات (مُخرَجات حقيقية من `BPETokenizer`):

```python
>>> from babytorch.text import BPETokenizer
>>> tok = BPETokenizer().fit("low lower lowest new newer newest",
...                          vocab_size=20)
>>> list(tok.merges.items())[:5]        # the first learned merge rules
[(('w', 'e'), 'we'),        # "we" occurs in lower/lowest/newer/newest...
 (('l', 'o'), 'lo'),        # ...so it merges first; then "lo",
 (('n', 'e'), 'ne'),        # then "ne",
 (('w', '</w>'), 'w</w>'),
 (('lo', 'we'), 'lowe')]    # merges of merges: pieces grow
>>> tok.encode("lowest")
[13, 16]
>>> [tok.inverse_vocab[i] for i in tok.encode("lowest")]
['lowe', 'st</w>']          # two subwords, not six characters
```

تفصيلان يستحقّان الملاحظة:

* **`</w>` علامة نهاية الكلمة** تُلحَق بكلّ كلمة قبل التدريب. تتيح للمُفردات أن
  تُميّز "low في نهاية كلمة" عن "low داخل *lowest*"، وتمنع عمليات الدمج من لصق
  كلمات مُنفصلة بعضها ببعض. وعند فكّ الترميز، تصير `</w>` مسافةً.
* **الترميز = إعادة تطبيق عمليات الدمج.** لتجزئة نصّ جديد، قسِّمه إلى محارف
  وطبِّق قواعد الدمج المُتعلَّمة بالترتيب الذي تعلَّمت به. تتقلّص الكلمات الشائعة
  إلى رموز مُفردة؛ أمّا الكلمة النادرة فتتفكّك بأناقة إلى قِطَع دون الكلمة
  (subword) — لا إلى `???` أبدًا.

هذه حقًّا الخوارزمية التي تقف خلف مُجزّئات GPT-2/3/4 (تبدأ مُجزّئاتها من
*البايتات* بدل المحارف وتُضيف تحسينًا كثيفًا، لكنّ حلقة الدمج هي نفسها). ومِقبض
`vocab_size` يُقايض طول المُتتالية بحجم المُفردات؛ وتستقرّ نماذج الإنتاج عند نحو
50,000–100,000 رمز.

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/text/tokenizers.py</code> (حلقة الدمج، والترميز بإعادة التطبيق)</summary>

```python
    def fit(self, text, vocab_size=512, verbose=False):
        """Learn merge rules from ``text`` up to ``vocab_size`` tokens.

        A special end-of-word marker ``</w>`` is appended to each word so
        the model can tell where words end (and doesn't merge across
        spaces).
        """
        # Represent each unique word as space-separated characters + </w>.
        words = text.split()
        sequences = Counter(' '.join(list(w) + ['</w>']) for w in words)

        # Seed the vocabulary with the base characters.
        base = set()
        for seq in sequences:
            base.update(seq.split())
        vocab = sorted(base)

        while len(vocab) < vocab_size:
            pair_counts = self._get_pair_counts(sequences)
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            sequences = self._merge_pair(best, sequences)
            self.merges[best] = ''.join(best)
            merged = ''.join(best)
            if merged not in vocab:
                vocab.append(merged)
    # ...
    def _tokenize_word(self, word):
        """Apply the learned merges to one word, return its sub-tokens."""
        symbols = list(word) + ['</w>']
        # Replay merges in the order they were learned (dict preserves it).
        for pair, merged in self.merges.items():
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    symbols[i:i + 2] = [merged]
                else:
                    i += 1
        return symbols
```

</details>

## من الرموز إلى أزواج التدريب — تسميات بلا مقابل

المُدوَّنة المُجزّأة مصفوفة واحدة طويلة من المُعرِّفات:

```python
data = np.array(tokenizer.encode(text))     # e.g. 1,115,394 ids for Shakespeare
```

والآن الجزء الجميل. احتاجت مُصنِّفات الفصل 3 إلى تسميات مصنوعة يدويًّا. أمّا
التنبّؤ بالرمز التالي فيحصل عليها **من النصّ الخام نفسه**: التسمية لأيّ موضع هي
ببساطة الرمز الذي جاء تاليًا فعلًا. اقتطع نافذة من `block_size` رمزًا للمُدخل
`x`، والنافذة نفسها مُزاحةً خطوةً واحدة إلى اليمين للهدف `y`:

![النصّ يصير أرقامًا والتسميات تأتي بلا مقابل: encode يُحوِّل النصّ إلى مُعرِّفات وdecode يُعيدها إلى نصّ؛ وفي الأسفل، نافذة x فوق المُدوَّنة ونافذة y مُزاحةً خطوةً واحدة إلى اليمين تجعلان كلّ موضع تمرينًا على الرمز التالي](../figures/fig-tokenization.svg)

نافذة واحدة طولها `T` تُنتج `T` من تمارين التنبّؤ دفعةً واحدة — الموضع 0 يتنبّأ
انطلاقًا من رمز سياق واحد، والموضع 1 من رمزين، وهكذا. تفعل `get_batch` في
[`tutorials/llm/common.py`](../../tutorials/llm/common.py) هذا بالضبط: اختر
`batch_size` من الإزاحات العشوائية، وكدِّس النوافذ في زوج `x, y` شكله `(B, T)`.

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>tutorials/llm/common.py</code></summary>

```python
def get_batch(data, block_size, batch_size):
    """Sample a random mini-batch of (context, next-token) pairs.

    We pick ``batch_size`` random start positions and cut ``block_size``
    tokens for the input ``x`` and the same window shifted one step to the
    right for the target ``y`` -- so ``y[t]`` is the token that really
    followed ``x[t]``.  Language modelling is next-token prediction, and
    this is where the "next token" labels come from, for free, from raw
    text.
    """
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y
```

</details>

هذا هو **الإشراف الذاتي**، وهو السبب في أنّ نماذج اللغة استطاعت أن تكبر إلى هذا
الحدّ: كلّ قصاصة نصّ على الأرض بيانات تدريب مُسبَقة التسمية لهدف "خمِّن الرمز
التالي". لا حاجة إلى مُوسِّمين.

فالنصّ الآن مُوَتِّرات من أعداد صحيحة، مع أهداف. لكنّ المُعرِّف الصحيح اسمٌ بلا
معنًى — فالمُعرِّف 13 ليس "أكبر من" المُعرِّف 12. وأوّل ما يفعله النموذج أن يضع
مكان كلّ مُعرِّف *متّجهًا* مُتعلَّمًا قادرًا على حمل المعنى (`Embedding` من الفصل
3). أمّا ما يحدث بعد ذلك — كيف يتسنّى للموضع 7 أن يستشير المواضع 0–6 قبل أن يُطلق
تخمينه — فتلك هي البنية المعمارية التي غيَّرت كلّ شيء.

## تمارين

**اختبر فهمك** (الإجابات تُفتَح بالنقر):

**س1.** ما قيمة `CharTokenizer("abcabc").vocab_size`؟

<details><summary>الإجابة</summary>

3. المُفردات هي *مجموعة* المحارف — `a` و`b` و`c` — بغضّ النظر عن تكرار كلٍّ منها.

</details>

**س2.** لماذا تُلحِق BPE علامة `</w>` بكلّ كلمة قبل تعلّم عمليات الدمج؟

<details><summary>الإجابة</summary>

مهمّتان: تمنع عمليات الدمج من لصق كلمات مُنفصلة بعضها ببعض، وتتيح للمُفردات أن
تُميّز "low في نهاية كلمة" عن "low داخل *lowest*" — رموز مختلفة، وإحصاءات مختلفة.
وعند فكّ الترميز، تصير `</w>` مسافةً.

</details>

**س3.** في نافذتَي x/y، ما الذي يُشرِف بالضبط على التنبّؤ عند الموضع `t` — ومن
أنتج تلك التسمية؟

<details><summary>الإجابة</summary>

`y[t]`، الرمز الذي *تلا فعلًا* `x[..t]` في المُدوَّنة. لم يُسمِّ أحد شيئًا: النصّ
الخام هو مفتاح إجابته بنفسه. ذلك هو الإشراف الذاتي، وهو سبب كون كلّ قصاصة نصّ
بيانات تدريب.

</details>

**طبِّقه بنفسك** — نفِّذ `WordTokenizer` مع مَخرج طوارئ `<unk>` (لتَلمس مشكلة
الخروج عن المُفردات بنفسك) و★ `most_frequent_pair` (دورة واحدة من ذراع BPE) في
[`exercises/ch05_tokenization.py`](../exercises/ch05_tokenization.py)، ثمّ شغّل
`pytest book/exercises/test_ch05_tokenization.py -v`.
([كيف تعمل التمارين](../exercises/README.md).)

---

**ملفّات المصدر لهذا الفصل:**
[`babytorch/text/tokenizers.py`](../../babytorch/text/tokenizers.py) (المُجزّئان) ·
[`babytorch/datasets/text.py`](../../babytorch/datasets/text.py) (Tiny Shakespeare) ·
[`tutorials/llm/common.py`](../../tutorials/llm/common.py) (`get_batch`) ·
[`tests/test_tokenizer.py`](../../tests/test_tokenizer.py)

[→ الفصل 4: التدريب](04-training.md) | [المحتويات](README.md) | [الفصل 6: الانتباه ←](06-attention.md)
