# الفصل 3 — الشبكات العصبية

*الجزء الأول، الفصل 3 من 4. يستطيع الاشتقاق التلقائي أن يشتقّ أيّ شيء نبنيه من
عمليات المُوَتِّرات. وهذا الفصل يدور حول ماذا نبني: الطبقات، والنماذج، ودوالّ
الخسارة التي تمنحها شيئًا لتتعلّمه.*

## النموذج دالّة ذات مقابض

كلّ ما يُنتجه `babytorch.nn` هو، رياضيًّا، مجرّد

```
prediction = f(input, parameters)
```

— دالّة تُحوِّل مُوَتِّرًا إلى مُوَتِّر، ولها بعض **المُعامِلات** (parameters):
مُوَتِّرات أُنشئت بـ `requires_grad=True`، وهي ما سيضبطه المُحسِّن (optimizer).
ويضمن الفصل 2 أنّه أيًّا كانت `f` التي نُركّبها، تأتي تدرُّجات كلّ مُعامِل مجّانًا.
فهذا الفصل في حقيقته فهرس لـ *أشكال مفيدة من f*، من طبقة عمل أساسية واحدة إلى
القطع التي سيحتاجها نموذج GPT.

## طبقة العمل الأساسية: `Linear`

الطبقة الأساسية تضرب دخلها بمصفوفة أوزان وتُضيف انحيازًا
([`babytorch/nn/nn.py`](../../babytorch/nn/nn.py)):

```python
y = x @ W + b        # (batch, in) @ (in, out) + (1, out) -> (batch, out)
```

كلّ عمود من أعمدة الخرج الـ `out` يحسب مجموعًا موزونًا لكلّ المدخلات الـ `in` —
أي `out` كواشف صغيرة، لكلٍّ منها حرّيّة أن يزن سمات الدخل بطريقته الخاصّة. ولاحظ
كم تنساب الدُّفعة (batch) عبرها بطبيعيّة: لـ `x` صفّ واحد لكلّ مثال، ويتولّى `@`
معالجتها جميعًا دفعةً واحدة، ويمنح البثّ (الفصل 1) كلّ صفّ الانحياز نفسه.

```python
import babytorch.nn as nn

layer = nn.Linear(3, 5)      # in_features=3, out_features=5
layer.w.shape                # (3, 5)
layer.b.shape                # (1, 5)
```

تفصيلة غير بديهيّة: تبدأ الأوزان أرقامًا *عشوائيّة صغيرة*، مسحوبة من `U(-k, k)`
حيث `k = 1/sqrt(in_features)`. عشوائيّة — لأنّها لو بدأت كلّها متساوية، لحسب كلّ
كاشف الشيء نفسه وتلقّى التدرُّج نفسه، إلى الأبد (لا شيء يجعلها تختلف قطّ). وصغيرة،
ومُقاسة على عرض الطبقة — كي يكون للمخرجات تباين مقارب لتباين المدخلات، فلا تنفجر
الإشارات ولا تندثر وهي تمرّ عبر طبقات كثيرة. وسوء تهيئة الأوزان من الطرق
الكلاسيكية التي تفشل بها الشبكات العميقة في التدريب فشلًا صامتًا.

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/nn/nn.py</code> (الطبقة كاملةً)</summary>

```python
    def __init__(self, in_features, out_features, activation_function=None):
        k = 1.0 / math.sqrt(in_features)
        self.w = Tensor(xp.random.uniform(-k, k, (in_features, out_features)),
                        requires_grad=True)
        self.b = Tensor(xp.random.uniform(-k, k, (1, out_features)),
                        requires_grad=True)
        self.activation_function = activation_function

    def forward(self, x):
        out = x @ self.w + self.b
        if self.activation_function:
            out = self.activation_function(out)
        return out
```

</details>

## لماذا نحتاج إلى اللاخطّيّات

كدّس طبقتين خطّيّتين فتحصل على... دالّة خطّية:
`(x @ W1) @ W2 = x @ (W1 @ W2)`. ينهار التركيب. ومهما بلغ عمق الكومة، فلن تستطيع
رسم سوى خطوط قرار مستقيمة.

الإصلاح يكلّف سطرًا واحدًا: أقحِم دالّة **لا خطّية** بسيطة بين الطبقات. ويوفّر
BabyTorch المجموعة القياسية، وحداتٍ ودوالَّ على المُوَتِّر في آنٍ معًا:

```
ReLU:  max(0, x)        cheap, sharp — the modern default
Tanh:  squash to (-1,1) smooth, classic
Sigmoid: squash to (0,1) turns a score into a probability
GELU:  smooth ReLU      what GPT uses (chapter 7)
```

بوجود اللاخطّيّات بينها، تستطيع الطبقات الخطّية المكدّسة أن تنحني — وكومة من
الانحناءات تستطيع تقريب أيّ دالّة معقولة. وتلك هي فكرة الشبكة العصبية بأكملها:

```python
model = nn.Sequential(
    nn.Linear(2, 8, nn.ReLU()),    # BabyTorch: activation as optional 3rd arg
    nn.Linear(8, 1, nn.Sigmoid()),
)
```

```
 x ──► Linear(2→8) ──► ReLU ──► Linear(8→1) ──► Sigmoid ──► prediction
        8 detectors     bend      combine        to (0,1)
```

## `Module`: الصنف الأساس الوحيد

تتشارك الطبقات والنماذج الكاملة صنفًا أساسيًّا صغيرًا، هو `Module`
([`babytorch/nn/nn.py`](../../babytorch/nn/nn.py)). وكتابة صنف جديد تتطلّب خطوتين
— خزِّن المُعامِلات بوصفها خصائص، ونفِّذ `forward`:

```python
class TinyModel(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear(2, 8)     # sub-module
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.layer2(self.layer1(x).relu())

model = TinyModel()
model(x)                      # calling the model calls forward()
```

وكلّ ما عدا ذلك موروث، وكلّه يعمل عبر **المرور على الخصائص**:

* `model.parameters()` — يجمع تعاوديًّا كلّ مُوَتِّر بـ `requires_grad=True` من
  الوحدة ووحداتها الفرعية وأيّ *قوائم* من الوحدات الفرعية (تلك الحالة الأخيرة هي
  كيفيّة العثور على كومة كتل المُحوِّل (Transformer)). لا نداءات تسجيل، ولا حِيَل
  أصناف وصفية (metaclass).
* `model.zero_grad()` — يُصفِّر كلّ تلك الـ `.grad` (شرح الفصل 2 لماذا تتراكم
  لولا ذلك عبر الدُّفعات).
* `model.train()` / `model.eval()` — يقلب راية `training` على كلّ وحدة فرعية؛
  فطبقات مثل Dropout تتصرّف على نحوٍ مختلف حسب الوضع.
* `model.save(path)` / `nn.Module.load(path, model)` — يحفظ المُعامِلات بـ pickle
  (بعد تحويلها إلى NumPy على المعالج، فيُحمَّل نموذجٌ دُرِّب على GPU في أيّ مكان).
* `model.num_parameters()` — كم رقمًا قابلًا للتدريب تملك.

و`Sequential`، المُستخدَم أعلاه، هو نفسه سبعة أسطر من `Module`: يخزّن الطبقات
المُعطاة في قائمة، ويُمرِّر `forward` خرج كلٍّ منها إلى التالية.

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/nn/nn.py</code> (المرور على الخصائص، وتصفير التدرُّج)</summary>

```python
    def parameters(self):
        """Collect every trainable tensor in this module, recursively.

        We look through the instance's attributes for:
        * tensors with ``requires_grad=True``  -> parameters of this module;
        * sub-modules                          -> ask them for theirs;
        * lists/tuples of sub-modules          -> same (e.g. Sequential,
          or the list of blocks in a Transformer).
        """
        params = []
        for value in vars(self).values():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params
    # ...
    def zero_grad(self):
        """Reset the gradients of all parameters.

        Call this after each optimizer step: ``backward()`` *accumulates*
        gradients, so leftovers from the previous batch would otherwise
        contaminate the next one.
        """
        for p in self.parameters():
            p.grad = None
```

</details>

## دوالّ الخسارة: تحويل "الخطأ" إلى رقم

يحتاج التدريب إلى عدد قياسي واحد يقول *كم يُخطئ* النموذج — وهو المُوَتِّر الذي
نستدعي عليه `.backward()`. ودالّتا خسارة تغطّيان معظم التعلّم العميق
([`babytorch/nn/loss.py`](../../babytorch/nn/loss.py)):

### `MSELoss` — للانحدار (التنبّؤ بالأرقام)

```
loss = mean( (prediction − target)² )
```

التربيع يجعل خطأ الاتّجاهين موجبًا، ويعاقب الأخطاء الكبيرة أشدّ بكثير من الصغيرة.

### `CrossEntropyLoss` — للتصنيف (الاختيار بين الفئات)

يُصدر النموذج درجة خامّة واحدة لكلّ فئة (**logits**). وتحوّل الإنتروبيا المتقاطعة
الدرجات إلى احتمالات بـ softmax، ثمّ تُحمِّل النموذج `−log p` حيث `p` هو الاحتمال
الذي أعطاه للفئة *الصحيحة*:

```
confident & right:  p ≈ 1   ->  −log p ≈ 0      (barely charged)
unsure:             p ≈ 0.5 ->  −log p ≈ 0.7
confident & wrong:  p ≈ 0   ->  −log p → ∞      (severely charged)
```

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)   # logits (batch, classes); targets: class ids
```

داخليًّا يستخدم `log_softmax` — دمج softmax واللوغاريتم عبر حيلة log-sum-exp، فلا
تطفح قيم logits الضخمة أو الضئيلة. **احفظ هذه الخسارة في ذهنك:** نموذج اللغة الذي
يتنبّأ بالرمز التالي هو بالضبط هذا التصنيف، مع `num_classes = vocabulary size`.
ودالّة الخسارة التي تُدرِّب BabyGPT في الفصل 8 هي هذه نفسها، دون تغيير.

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/nn/loss.py</code></summary>

```python
    def forward(self, predictions, targets):
        assert isinstance(predictions, Tensor), "predictions must be a Tensor"

        # Targets may arrive as a Tensor, a list, or an array; index arrays
        # must be integers.
        if isinstance(targets, Tensor):
            targets = targets.data
        targets = xp.asarray(targets).astype(xp.int64)
        assert targets.ndim == 1, (
            f"targets must be a 1-D array of class ids, got shape {targets.shape}")
        n = targets.shape[0]

        log_probs = predictions.log_softmax(axis=-1)      # (n, num_classes)
        # Pick out, for every row, the log-probability of its true class.
        picked = log_probs[xp.arange(n), targets]          # (n,)
        return -picked.mean()
```

</details>

## الطبقات المتخصّصة

أربع طبقات أخرى، تُشرَح كلٌّ منها شرحًا وافيًا حين يحتاجها الجزء الثاني:

* **`Embedding(num, dim)`** — جدول بحث قابل للتعلّم: المُعرِّف الصحيح `i` →
  الصفّ `i`، أي متّجه بطول `dim`. التمرير الأمامي مجرّد فهرسة؛ والاشتقاق الخلفي
  للشرائح في الاشتقاق التلقائي (autograd) ينثر التدرُّجات على الصفوف التي
  استُخدمت. وهكذا تدخل الرموز (tokens) نموذجَ اللغة (الفصل 5).
* **`LayerNorm(features)`** — يُعيد قياس كلّ متّجه سمات إلى متوسّط صفري وتباين
  واحدي، ثمّ يترك مُعامِلين متعلَّمين (`gamma` و`beta`) يُلغيان ذلك حيث يكون
  مفيدًا. إنّه المُثبِّت الذي يُبقي المُحوِّلات العميقة قابلة للتدريب (الفصل 7).
* **`Dropout(p)`** — أثناء التدريب فقط، يُصفِّر عشوائيًّا نسبةً `p` من القيم
  (تُضخَّم الناجيات بـ `1/(1−p)`، فيمكن أن يكون التقييم بلا عمل). ولأنّ أيّ قيمة
  قد تتلاشى، لا تستطيع الشبكة الاتّكاء على مسار هشّ واحد — علاج فظّ لكنّه فعّال
  للإفراط في المُلاءمة.
* **`Conv2D` / `MaxPool2D` / `Flatten`** — منعطف الرؤية: أزلِق مُرشِّحات متعلَّمة
  فوق صورة، وخفِّض الدقّة، وسطِّح تمهيدًا لطبقة خطّية أخيرة. لا حاجة إليها في GPT،
  لكنّ درس MNIST يستخدمها، ويحوي
  [`operations.py`](../../babytorch/engine/operations.py) نسختين معًا: التفافًا
  (convolution) مقروءًا حلقةً بحلقة، والنسخة السريعة `im2col` التي تحوّل الالتفاف
  إلى عملية ضرب مصفوفات واحدة كبيرة.

أتُفضّل الدوالّ على كائنات الطبقات؟ يوفّر `babytorch.nn.functional` (يُستورَد
عادةً باسم `F`) الدوالّ `F.relu` و`F.softmax` و`F.cross_entropy`، ... —
الرياضيّات نفسها دون بناء وحدات.

![نموذج ذو مقابض: يمرّ x عبر Linear من 2 إلى 8، ثمّ Tanh، ثمّ Linear من 8 إلى 1، ثمّ Sigmoid ليُنتج تنبّؤًا، تقارنه MSELoss بالأهداف لتُخرج الرقم الواحد الذي تستدعي عليه backward](../figures/fig-mlp.svg)

**جرِّبه**

```python
>>> import babytorch, babytorch.nn as nn
>>> model = nn.Sequential(nn.Linear(2, 8, nn.Tanh()), nn.Linear(8, 1, nn.Sigmoid()))
>>> model.num_parameters()          # (2*8 + 8) + (8*1 + 1) = 33
33
>>> x = babytorch.randn(4, 2)       # a batch of 4 two-feature examples
>>> model(x).shape
(4, 1)
>>> for name, p in model.named_parameters():
...     print(name, p.shape)
layers.0.w (2, 8)
layers.0.b (1, 8)
layers.1.w (8, 1)
layers.1.b (1, 1)
```

النموذج موجود ويُحوِّل المدخلات — لكنّ أوزانه عشوائية، فتنبّؤاته ضوضاء. وجعلها
*ليست* ضوضاء هو موضوع الفصل 4.

## تمارين

**اختبر فهمك** (الإجابات تُفتَح بالنقر):

**س1.** كم مُعامِلًا يملك `nn.Linear(128, 10)`؟

<details><summary>الإجابة</summary>

1290: مصفوفة أوزان `(128, 10)` زائد انحياز `(1, 10)` — 1280 + 10.

</details>

**س2.** ما الذي تستطيع كومة من خمس طبقات `Linear` *بلا دوالّ تنشيط* حسابَه ولا
تستطيعه طبقة `Linear` واحدة؟

<details><summary>الإجابة</summary>

لا شيء. `(x @ W1) @ W2 = x @ (W1 @ W2)` — تنهار الكومة إلى مصفوفة واحدة. ولا
يؤتي العمق ثماره إلّا حين تجلس دالّة لا خطّية بين الطبقات.

</details>

**س3.** يمنح مُصنِّف الفئةَ الصحيحة احتمالًا `p = 0.01`. كم تُحمِّل الإنتروبيا
المتقاطعة تقريبًا لذلك المثال، وما الدرس؟

<details><summary>الإجابة</summary>

`−ln(0.01) ≈ 4.6` — مقابل ≈ 0.1 لإجابة صحيحة واثقة. تعاقب الإنتروبيا المتقاطعة
*الخطأ الواثق* بوحشية، وهو بالضبط الضغط الذي يجعل النموذج يُعايِر احتمالاته.

</details>

**طبِّقه بنفسك** — نفِّذ `RMSNorm` (تسوية LLaMA — تشغل خانة LayerNorm في الفصل 7،
لكنّها أنحف) و★ `bce_loss` في [`exercises/ch03_nn.py`](../exercises/ch03_nn.py)،
ثمّ شغّل `pytest book/exercises/test_ch03_nn.py -v`. عمليات المُوَتِّر فقط: إن
بنيتَه على نحوٍ صحيح، فالتمرير الخلفي مجّاني.
([كيف تعمل التمارين](../exercises/README.md).)

---

**ملفّات المصدر لهذا الفصل:**
[`babytorch/nn/nn.py`](../../babytorch/nn/nn.py) (صنف Module وكلّ الطبقات) ·
[`babytorch/nn/loss.py`](../../babytorch/nn/loss.py) (دوالّ الخسارة) ·
[`babytorch/nn/functional.py`](../../babytorch/nn/functional.py) (الصيغ الدالّية)

[→ الفصل 2: الاشتقاق التلقائي](02-autograd.md) | [المحتويات](README.md) | [الفصل 4: التدريب ←](04-training.md)
