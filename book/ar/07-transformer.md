# الفصل 7 — المُحوِّل

*الجزء الثاني، الفصل 3 من 4. اكتمل بناء الانتباه. والآن نجمعه في البنية التي
تقف خلف GPT-2 وGPT-4 وLLaMA — ونرى لماذا يمكن أصلًا تدريب مكدّس منه بعمق 24
طبقة.*

## النصف الآخر: شبكة MLP

ينقل الانتباه المعلومات *بين* المواضع. وتتبعه شبكةٌ صغيرة من طبقتين تُطبَّق *على
كلّ موضع على حِدة* — الأوزان نفسها لكلّ موضع، دون أيّ تخاطب بينها
([`tutorials/llm/model.py`](../../tutorials/llm/model.py)):

```python
class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        self.fc = nn.Linear(n_embd, 4 * n_embd)    # widen 4x
        self.gelu = nn.GELU()                      # bend
        self.proj = nn.Linear(4 * n_embd, n_embd)  # narrow back
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.gelu(self.fc(x))))
```

تقسيمٌ مفيد للعمل (وإن كان غير دقيق): **الانتباه يجمع، وشبكة MLP تهضم.** بعد أن
يكون موضعٌ ما قد استحضر «كان هناك قطّ، قبل 13 رمزًا»، فإنّ شبكة MLP هي المكان
الذي تُعالَج فيه تلك المعلومة إلى شيء تستطيع الطبقة التالية استخدامه. والتوسيع
المؤقّت إلى `4·C` (النسبة القياسية في المُحوِّل) يمنحها متّسعًا للحساب؛ وGELU هي
صيغة ReLU الملساء من الفصل 3، خيار عائلة GPT.

## الكتلة، والحيلتان اللتان تجعلان العمق ممكنًا

ليست الكتلة (block) الواحدة في المُحوِّل سوى الطبقتين الفرعيّتين موصولتين معًا —
لكنّ طريقة التوصيل هي كلّ شيء:

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # communicate
        x = x + self.mlp(self.ln2(x))    # compute
        return x
```

```
   x ──┬────────────────────────►(+)──┬────────────────────►(+)──► out
       │                          ▲   │                      ▲
       └─► LayerNorm ─► attention ┘   └─► LayerNorm ─► MLP ──┘
```

**الحيلة 1 — الوصلات المتبقّية (residual).** *يُضاف* خرجُ الطبقة الفرعية إلى
دخلها، لا أن يحلّ محلّه. ولهذا نتيجتان. أمّا في التعلّم: فلا تحتاج الكتلة إلّا
إلى إنتاج *تعديل* مفيد على `x`، لا إلى إعادة بناء التمثيل بأكمله — وإن لم يكن
لديها ما تضيفه، فإنّ خيار «ألّا تفعل شيئًا» متاح بلا عناء. وأمّا في التدرُّجات:
فيقول الفصل 2 إنّ الجمع يُمرِّر التدرُّجات دون تغيير، فتُشكّل سلسلة عمليات `+`
طريقًا سريعًا غير منقطع من الخسارة رجوعًا مباشرةً إلى الطبقة 1. كدّس 20 طبقة
عادية، فيتقلّص مقدار الإشارة (أو ينفجر) أُسّيًّا بفعل الضرب المتكرّر للمشتقّات
المحلّية؛ والطريق السريع المتبقّي هو ما يجنّب المُحوِّلات العميقة ذلك المصير.

**الحيلة 2 — التسوية القَبْلية (pre-LayerNorm).** ترى كلُّ طبقة فرعية نسخةً
مُسوّاة من `x` (تسوية الطبقة (LayerNorm) من الفصل 3: متوسّط صفري، وتباين مقداره
1 لكلّ موضع، ثمّ إعادة قياس مُتعلَّمة). وفي الوقت نفسه، يُراكم التيّار المتبقّي
ذاته مجاميعَ مخرجاتِ طبقات فرعية كثيرة، ولولا التسوية لانجرف مقياسه مع ازدياد
العمق؛ فالتسوية *عند مدخل كلّ طبقة فرعية* تُبقي كلّ طبقة تعمل على مُدخلات لها
المقدار نفسه، سواء أكانت الطبقة 1 أم الطبقة 24.

لا تضيف أيٌّ من الحيلتين قدرةً تعبيرية. وكلتاهما موجودة لسبب واحد: *قابلية
التدريب*. فالابتكار الحقيقي في المُحوِّل ليس الانتباه وحده — بل بنيةٌ تنجو
تدرُّجاتها من العمق.

## النموذج بأكمله

`GPT` هو تضمينات، ومكدّس من الكتل، ورأس قراءة (readout):

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4,
                 n_layer=4, dropout=0.0):
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = [Block(n_embd, n_head, block_size, dropout)
                       for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
```

![بنية BabyGPT الكاملة، تُقرأ من الأسفل إلى الأعلى: تدخل معرّفات الرموز إلى تضمينات الرموز وتضمينات المواضع، فتُجمَع ثمّ يُطبَّق عليها الإسقاط؛ وتُطبِّق ستّ كتل مُحوِّل متطابقة كلٌّ منها تسويةَ الطبقة، والانتباهَ الذاتي السببي، وجمعًا متبقّيًا، وتسويةَ الطبقة، وشبكةَ MLP، وجمعًا متبقّيًا آخر؛ وتُنتج تسويةٌ نهائية للطبقة ورأسُ الخرج الخطّي (Linear) قيمَ logits بالشكل B في T في 65 — مع بطاقة نموذج لإعدادات train.py الافتراضية على اليمين](../figures/fig-babygpt.svg)

لاحظ ما هو الخرج: ليس تنبّؤًا واحدًا، بل تنبّؤًا **عند كلّ موضع** — فقيم logits
عند الموضع `t` هي تخمينه للرمز `t+1`، بعد أن انتبه إلى المواضع `0..t` فقط. وكلّ
التخمينات `T` تتدرّب على التوازي.

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>tutorials/llm/model.py</code> (التمرير الأمامي كاملًا دون اختصار)</summary>

```python
    def forward(self, idx, kv_caches=None):
        """idx: integer array (B, T) of token ids -> logits (B, T, vocab_size).

        ``kv_caches`` (generation only): one cache per block, carrying the
        keys/values of already-processed positions, so ``idx`` needs to
        hold only the tokens that come after them.  See ``generate``.
        """
        if isinstance(idx, Tensor):
            idx = idx.data
        idx = xp.asarray(idx).astype(xp.int64)
        B, T = idx.shape

        # With a cache, this call's tokens sit *after* the cached positions,
        # so their position ids start where the cache ends.
        past = 0
        if kv_caches is not None and kv_caches[0]["k"] is not None:
            past = kv_caches[0]["k"].shape[2]
        assert past + T <= self.block_size, (
            f"sequence length {past + T} exceeds block size {self.block_size}")

        tok = self.token_embedding(idx)                       # (B, T, C)
        pos = self.position_embedding(xp.arange(past, past + T))  # (T, C)
        x = self.drop(tok + pos)                              # broadcast add

        for i, block in enumerate(self.blocks):
            x = block(x, kv_caches[i] if kv_caches is not None else None)
        x = self.ln_f(x)
        return self.head(x)                                   # (B, T, vocab_size)
```

(`kv_caches` هو مخبّأ KV الخاص بالتوليد — حيلة للسرعة، لا جزء من البنية.
تجاهله في القراءة الأولى: فمن دون مخبّأ، تكون `past` صفرًا ويطابق هذا تمامًا
الشرح أعلاه. ويوضّح الفصل 8 ما الذي يمنحه المخبّأ.)

</details>

تستحقّ قائمةُ الكتل البايثونية العادية ملاحظةً: فدالّة `Module.parameters()` من
الفصل 3 تمرّ على قوائم السمات أيضًا، فتُكتشَف مُعامِلات المكدّس تلقائيًّا، ويصل
إليها جميعًا `loss.backward()`. ولا حاجة إلى أيّ عناء مع `ModuleList`.

## الخسارة هي خسارة الفصل 3 نفسها، دون تغيير

```python
def loss(self, idx, targets):
    logits = self.forward(idx)              # (B, T, V)
    logits = logits.reshape(B * T, V)       # every position = one example
    targets = targets.reshape(B * T)
    return nn.CrossEntropyLoss()(logits, targets)
```

سطِّح الدُّفعة والزمن معًا، فيغدو التنبّؤ بالرمز التالي *هو* التصنيف عينه بعدد
أصناف `V = vocab_size` — وهي الخسارة نفسها من الفصل 3، تُغذّى بـ `B·T` مثالًا
في كلّ خطوة. وهذا التكافؤ هو بيت القصيد في الكتاب بأكمله: **نموذج GPT مُصنِّفٌ
داخل حلقة.**

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>tutorials/llm/model.py</code> (النسخة الحقيقية، مع تفاصيل معالجة الأنواع)</summary>

```python
    def loss(self, idx, targets):
        """Cross-entropy between predictions and the shifted targets.

        ``targets`` is ``(B, T)``: for each position, the id of the token
        that actually came next.  We flatten batch and time together so the
        loss is one big classification over all positions at once.
        """
        logits = self.forward(idx)                      # (B, T, vocab)
        B, T, V = logits.shape
        logits = logits.reshape(B * T, V)
        if isinstance(targets, Tensor):
            targets = targets.data
        targets = xp.asarray(targets).astype(xp.int64).reshape(B * T)
        return nn.CrossEntropyLoss()(logits, targets)
```

</details>

## الحجم: أين تسكن المُعامِلات

الإعداد الافتراضي في الدرس التطبيقي (`train.py`: `n_embd=192` و`n_head=6`
و`n_layer=6` و`block_size=128`، ومُفردات محرفية ≈ 65) يبلغ وزنه نحو **2.7
مليون مُعامِل**:

| العنصر | العدد | الحصّة |
|-------|-------|-------|
| 6 كتل (انتباه + MLP + تسويات) | ≈ 2,669,000 | ~98% |
| تضمينات المواضع (128 × 192) | 24,576 | |
| تضمينات الرموز (65 × 192) | 12,480 | |
| التسوية النهائية + رأس الخرج | ≈ 12,900 | |

كلّ شيء تقريبًا يقع في الكتل، موزّعًا بنسبة 1:2 تقريبًا بين أوزان الانتباه وأوزان
MLP — وكلّ واحدة من تلك المصفوفات موجودة لتخدم `@`. ولنموذج GPT-2 الهيكل نفسه
عند `n_embd=768, n_layer=12` (124 مليون)؛ ولنموذج GPT-3 عند
`n_embd=12288, n_layer=96` (175 مليار). المقابض وما تمنحه:

* `n_embd` — عرض متّجه كلّ رمز؛ سعة التيّار.
* `n_layer` — العمق؛ مزيدٌ من جولات التواصل ثمّ الحساب.
* `n_head` — كم نمط انتباه في كلّ طبقة (يجب أن يقسم `n_embd`).
* `block_size` — أقصى طول للسياق؛ تكلفة الانتباه تنمو تناسبًا مع T².
* `dropout` — تنظيم، للحالة التي يفوق فيها النموذجُ حجمَ بياناته.

**جرِّبه**

```python
>>> import sys; sys.path.insert(0, "tutorials/llm")
>>> from model import GPT
>>> import babytorch
>>> model = GPT(vocab_size=65, block_size=32, n_embd=64, n_head=4, n_layer=2)
>>> model.num_parameters()
110529
>>> ids = (babytorch.rand(2, 10).data * 65).astype(int)   # 2 sequences, 10 tokens
>>> model(ids).shape
(2, 10, 65)                       # per position, a score for every token
```

الأوزان عشوائية، فالدرجات مجرّد ضوضاء — والنموذج يُهذي. والفصل 8 يحوّل هذا الهذيان
إلى شكسبير.

## تمارين

**اختبر فهمك** (الإجابات تُفتَح بالنقر):

**س1.** أيّ طبقة فرعية تتيح للموضع 7 أن يستخدم معلومات من الموضع 2 — الانتباه أم
MLP؟

<details><summary>الإجابة</summary>

الانتباه وحده. فشبكة MLP تُطبِّق الشبكة الصغيرة نفسها على كلّ موضع على حِدة —
ولا تنظر جانبًا أبدًا. ومن هنا يأتي إيقاع الكتلة: التواصل (الانتباه)، ثمّ الحساب
(MLP).

</details>

**س2.** تُضاعف `block_size`. أيّ عدد من المُعامِلات يتغيّر، وما الذي ينمو *أيضًا*
رغم أنّ أيّ أوزان لا تزيد؟

<details><summary>الإجابة</summary>

جدولُ تضمين المواضع (`block_size × n_embd`) وحده هو ما يكسب مُعامِلات. لكنّ *حساب
الانتباه وذاكرته* ينموان نموًّا تربيعيًّا — فجدول الدرجات `(T, T)` يصير أربعة
أضعاف. طول السياق باهظٌ في العمل، لا في الأوزان.

</details>

**س3.** أزل الوصلات المتبقّية، فيتوقّف نموذجٌ من 12 كتلة عن التدرّب. أيّ حقيقة من
الفصل 2 تفسّر ذلك؟

<details><summary>الإجابة</summary>

قاعدة السلسلة *تضرب* المشتقّات المحلّية على امتداد المسار — و12 طبقة فرعية
مكدّسة تضرب 12 عاملًا، فيتلاشى الحاصل أو ينفجر. أمّا الجمع فيُمرِّر التدرُّجات دون
تغيير، فتغدو السلسلة المتبقّية طريقًا سريعًا غير منقطع من الخسارة إلى الطبقة 1.

</details>

**طبِّقه بنفسك** — اكتب `count_gpt_parameters` بصيغة مغلقة (يختبرك المُصحِّح على
نماذج حقيقية — ولاحظ أنّ `n_head` لا يظهر أبدًا في صيغتك)، ثمّ ★ أجرِ عملية
جراحية حقيقية: `TiedGPT`، حيلة ربط الأوزان في GPT-2، في
[`exercises/ch07_transformer.py`](../exercises/ch07_transformer.py)؛ ثمّ شغّل
`pytest book/exercises/test_ch07_transformer.py -v`.
([كيف تعمل التمارين](../exercises/README.md).)

---

**ملفّات المصدر لهذا الفصل:**
[`tutorials/llm/model.py`](../../tutorials/llm/model.py) (`MLP` و`Block` و`GPT`) ·
[`babytorch/nn/nn.py`](../../babytorch/nn/nn.py) (`LayerNorm` و`Embedding` و`GELU`) ·
[`tests/test_training.py`](../../tests/test_training.py) (نموذج GPT صغير ثبت أنّه يتعلّم)

[→ الفصل 6: الانتباه](06-attention.md) | [المحتويات](README.md) | [الفصل 8: تدريب نموذج GPT ←](08-training-a-gpt.md)
