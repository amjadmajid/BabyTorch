# الفصل 2 — الاشتقاق التلقائي

*الجزء الأول، الفصل 2 من 4. هذا هو قلب كلّ إطار عمل للتعلّم العميق. بعد هذا
الفصل، لا يكون `loss.backward()` سحرًا — بل تسجيلًا يُعاد تشغيله بالمقلوب.*

## لماذا التدرُّجات؟

الشبكة العصبية دالّة فيها ملايين الأرقام القابلة للضبط (الأوزان)، والتدريب يعني
أن نُصغِّر رقمًا واحدًا — **الخسارة** (loss)، أي «كم نحن مخطئون؟». والسبيل العملي
الوحيد لتحسين ملايين الأوزان دفعةً واحدة هو أن نسأل، عن كلٍّ منها:

> إذا دفعتُ هذا الوزن إلى الأعلى قليلًا، فهل ترتفع الخسارة أم تنخفض، وبأيّ حدّة؟

وهذا بالضبط **مشتقّة**: `dLoss/dweight`، أي التدرُّج (gradient). ومتى عرف كلّ
وزن تدرُّجه، صارت الوصفة بسيطة إلى حدّ الإحراج — حرِّك كلّ وزن خطوةً صغيرة في
الاتّجاه الذي يُخفِّض الخسارة (تلك الوصفة هي الفصل 4). أمّا الجزء الصعب فهو حساب
مليون مشتقّة، بثمن زهيد وبدقّة تامّة. وتلك مهمّة الاشتقاق التلقائي (autograd)،
وهو يقوم على مبرهنة واحدة وبنية بيانات واحدة.

## المبرهنة: قاعدة السلسلة

الحسابات الكبيرة تراكيب من خطوات صغيرة. إذا كان `y = f(g(x))`، فإنّ

```
dy/dx  =  dy/dg  ·  dg/dx
```

معدّلات التغيّر تتضاعف عبر السلسلة. فإذا عرفنا المشتقّة *المحلّية* لكلّ خطوة صغيرة
— مجرّد `+` و`*` و`matmul` و`exp` وبضع عشرات غيرها — أمكننا ضربها معًا على امتداد
المسار من الخسارة رجوعًا إلى أيّ وزن. ما من خطوة معقّدة قطّ؛ كلّ ما هنالك أنّها
كثيرة.

## بنية البيانات: مخطّط الحساب

انتهى الفصل 1 بثلاثة حقول إضافية على كلّ مُوَتِّر (tensor). وهذا ما تشتريه لنا.
شغِّل:

```python
import babytorch
x = babytorch.tensor([2.0], requires_grad=True)
y = x * x + 3.0 * x            # y = x² + 3x  ->  y = 10
```

لم يحسب BabyTorch الرقم `10` فحسب. فكلّ عملية أنشأت مخرجها عبر
`Tensor._make_output`، الذي يخزّن رابطًا إلى العملية، والعملية احتفظت بمراجع إلى
مُدخلاتها. والنتيجة مخطّط:

![مخطّط الحساب للدالّة y = x² + 3x عند x = 2: الأسهم المتّصلة تحسب القيم وتسجّلها أماميًّا؛ والأسهم الحمراء المتقطّعة تعيد تشغيل التسجيل بالمقلوب، فتضرب المشتقّات المحلّية — وحيث يلتقي مساران عند x، تتجمّع التدرُّجات لتبلغ 7](../figures/fig-autograd.svg)

كلّ صندوق يعرف كيف يحسب مشتقّته المحلّية. وللحصول على `dy/dx`، سِر في المخطّط
**رجوعًا من y**، ضاربًا المشتقّات المحلّية كما تقتضي قاعدة السلسلة، وجامعًا حيث
تندمج المسارات (فـ x يسهم في y عبر مسارَين *اثنين* — وكلاهما يُحتسب):

```python
y.backward()
x.grad          # [7.]   because dy/dx = 2x + 3 = 7 at x = 2
```

وبقيّة هذا الفصل مجرّد: ما الذي في الصندوق، وكيف يعمل المسير الخلفي. وكلاهما قصير.

## داخل الصندوق: عقد `Operation`

تسكن الرياضيات في
[`babytorch/engine/operations.py`](../../babytorch/engine/operations.py).
كلّ عملية تنفّذ دالّتين:

* `forward` — يحسب المخرَج من المُدخلات (ويتذكّر المُدخلات؛ فالتمرير الخلفي سيحتاج
  إليها)؛
* `backward(grad)` — يستقبل `dLoss/dOutput` ويعيد `dLoss/dInput` لكلّ مُدخل.
  **تطبيق واحد لقاعدة السلسلة، لا أكثر.**

وهذا هو الضرب، كاملًا غير مختصر:

```python
class MulOperation(Operation):
    def forward(self, a, b):
        self.a = a
        self.b = b
        return a.data * b.data

    def backward(self, grad):
        # Product rule: d(a*b)/da = b  and  d(a*b)/db = a.
        a_grad = Operation.sum_to_shape(grad * self.b.data, self.a.shape)
        b_grad = Operation.sum_to_shape(grad * self.a.data, self.b.shape)
        return a_grad, b_grad
```

اقرأ `backward` مرّة واحدة بتمهّل: حساسيّة الخسارة تجاه `a` هي الحساسيّة الواردة
`grad` مضروبةً في المشتقّة المحلّية `b`. تلك هي قاعدة السلسلة، بالشيفرة، لعملية
واحدة. وكلّ عملية أخرى في الملفّ لها الشكل نفسه — لا يتغيّر سوى المشتقّة المحلّية:

| العملية | القاعدة المحلّية في `backward` |
|-----------|--------------------------|
| `a + b` | يمرّ التدرُّج دون تغيير إلى كليهما |
| `a * b` | حيلة التبديل: `a` يأخذ `grad·b`، و`b` يأخذ `grad·a` |
| `a ** n` | قاعدة القوّة: `grad · n·aⁿ⁻¹` |
| `exp(a)` | `grad · exp(a)` (مشتقّته هي نفسه — أعِد استخدام المخرَج المحفوظ) |
| `relu(a)` | مرِّر التدرُّج حيث `a > 0`، وصفرًا فيما عدا ذلك |
| `sum` | ابثّ التدرُّج رجوعًا إلى كلّ عنصر |
| `max` | وجِّه التدرُّج إلى الفائز وحده |
| `a[idx]` | بعثِر التدرُّج رجوعًا إلى المواضع التي قُرِئت |

وأمّا التي تستحقّ الحفظ، لأنّ المُحوِّلات (Transformers) مصنوعة منها — فهي ضرب
المصفوفات `C = A @ B`:

```
dL/dA = dL/dC @ Bᵀ           dL/dB = Aᵀ @ dL/dC
```

(أقنِع نفسك بالأشكال: فهي لا تتلاءم إلّا على نحوٍ واحد.)

## فصل الاهتمامات

لاحظ ما *لا* يعرفه `MulOperation`: أيّ شيء عن المخطّطات أو `requires_grad` أو
اجتياز `backward()`. فالعمليات تُجري الرياضيات على مصفوفات خام. وكلّ مسك الدفاتر
يسكن في
[`babytorch/engine/tensor.py`](../../babytorch/engine/tensor.py)، حيث كلّ دالّة
في Tensor ثلاثة أسطر من الغِراء:

```python
def __mul__(self, other):
    op = MulOperation()
    result = op.forward(self, other)
    return self._make_output(op, result, ..., "*")   # <- links the graph edge
```

*الرياضيات في `operations.py`، ومسك الدفاتر في `tensor.py`* — استحضر هذا الفصل،
ويتّسع المحرّك كلّه في رأسك.

## المسير الخلفي

`Tensor.backward()` (في
[`tensor.py`](../../babytorch/engine/tensor.py)) يفعل ثلاثة أشياء:

**1. البذرة.** تدرُّج الخسارة بالنسبة إلى نفسها يساوي 1:
`self.grad = ones_like(self.data)`.

**2. الترتيب الطوبولوجي** للمخطّط — أدرِج كلّ مُوَتِّر بحيث يظهر كلٌّ منها *بعد*
كلّ المُوَتِّرات التي يعتمد عليها. ومسيرٌ بالعمق أوّلًا من الخسارة ينجزه في أسطر
قليلة:

```python
def build_topo(v):
    if v not in visited:
        visited.add(v)
        if v.operation:
            for tensor in v.operation.inputs():
                build_topo(tensor)
        topo.append(v)
```

**3. إعادة التشغيل بالمقلوب.** سِر في تلك القائمة رجوعًا — من الخسارة نحو
المُدخلات — ودع كلّ عملية تحوّل تدرُّج مخرجها إلى تدرُّجات مُدخلاتها:

```python
for v in reversed(topo):
    if v.operation:
        grads = v.operation.backward(v.grad)
        for tensor, tensor_grad in zip(v.operation.inputs(), grads):
            if tensor.requires_grad:
                if tensor.grad is None:
                    tensor.grad = tensor_grad
                else:
                    tensor.grad = tensor.grad + tensor_grad   # accumulate!
```

الترتيب المعكوس يضمن أنّنا حين نطلب من عملية أن تدفع التدرُّجات إلى مُدخلاتها، يكون
تدرُّج مخرجها قد اكتمل بالفعل. وتلك هي الخوارزمية بأكملها — *الانتشار العكسي
(backpropagation) ترتيبٌ طوبولوجي زائد قاعدة السلسلة.*

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/engine/tensor.py</code> (كامل <code>backward()</code>، غير مختصر)</summary>

```python
    def backward(self, grad=None):
        """Run backpropagation from this tensor through the whole graph.

        Typically called on a scalar loss::

            loss.backward()

        After it returns, every tensor with ``requires_grad=True`` that
        contributed to ``loss`` holds ``dloss/dtensor`` in its ``.grad``.

        How it works, in three steps:

        1. Seed the output gradient: ``dloss/dloss = 1``.
        2. Sort the graph so every tensor comes *after* everything it
           depends on (a *topological sort*).
        3. Walk that order in reverse -- from the loss back to the
           inputs -- asking each operation to convert its output gradient
           into input gradients (chain rule), and *accumulating* them
           (a tensor used in several places sums the gradients from all
           of its uses).
        """
        if self.grad is None:
            if grad is not None:
                grad = xp.array(grad, dtype=self.data.dtype)
                assert grad.shape == self.data.shape, (
                    f"backward() gradient shape {grad.shape} must match "
                    f"tensor shape {self.data.shape}")
                self.grad = grad
            else:
                self.grad = xp.ones_like(self.data)

        if not self.requires_grad:
            return

        # -- step 2: topological sort ----------------------------------
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                if v.operation:
                    for tensor in v.operation.inputs():
                        build_topo(tensor)
                topo.append(v)

        build_topo(self)

        # -- step 3: chain rule in reverse order ------------------------
        for v in reversed(topo):
            if v.operation:
                grads = v.operation.backward(v.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)

                for tensor, tensor_grad in zip(v.operation.inputs(), grads):
                    if tensor.requires_grad:
                        if tensor.grad is None:
                            tensor.grad = tensor_grad
                        else:
                            tensor.grad = tensor.grad + tensor_grad
```

</details>

**لماذا `+=` لا `=`؟** المُوَتِّر المستخدَم في مواضع عدّة يستقبل تدرُّجًا من كلّ
استخدام، وقاعدة السلسلة تقول إنّ الإسهامات على امتداد المسارات المختلفة **تُجمَع**.
في مثالنا يُغذّي `x` كلًّا من `x*x` و`3*x`؛ وفي المُحوِّل، تخدم مصفوفة أوزان واحدة
كلّ موضع في الدُّفعة. والتراكم أيضًا هو سبب وجوب أن تنادي حلقات التدريب `zero_grad()`
بين الدُّفعات — وإلّا لظلّت البقايا تتراكم (الفصل 4).

## دَيْن البثّ: `sum_to_shape`

لبثّ الفصل 1 نتيجة في التمرير الخلفي. إذا نُسِخ انحياز شكله `(1, 10)` نسخًا
افتراضيًّا عبر 32 صفًّا في الطريق أمامًا، فإنّ كلًّا من الصفوف الـ 32 يشعر بأثر
الانحياز — فعند العودة، يكون تدرُّج الانحياز **مجموع** تدرُّجات الصفوف الـ 32 كلّها.
والمُساعِد `Operation.sum_to_shape` يُلغي كلّ بثّ على هذا النحو، وقد رأيته
مُستدعًى في `MulOperation.backward` أعلاه. القاعدة الذهبية:

```
broadcast (copy) in the forward pass   <=>   sum in the backward pass
```

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/engine/operations.py</code></summary>

```python
    @staticmethod
    def sum_to_shape(grad, shape):
        """Undo broadcasting: reduce ``grad`` back to ``shape`` by summing.

        If the forward pass broadcast a tensor of ``shape`` up to
        ``grad.shape``, each original element was copied into several
        output positions.  By the chain rule its gradient is the *sum* of
        the gradients at all those positions.

        Two things may have happened during broadcasting, undone in order:

        1. extra dimensions were prepended  -> sum them away entirely;
        2. size-1 dimensions were stretched -> sum them back to size 1.

        Example: ``bias`` of shape ``(1, 10)`` added to a batch of shape
        ``(32, 10)`` receives a ``(32, 10)`` gradient, which is summed
        over axis 0 back to ``(1, 10)``.
        """
        # 1) sum away prepended dimensions
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        # 2) sum stretched dimensions back to size 1
        axes = tuple(i for i, dim in enumerate(shape)
                     if dim == 1 and grad.shape[i] != 1)
        if axes:
            grad = grad.sum(axis=axes, keepdims=True)
        return grad
```

</details>

## ثِق، ولكن تحقّق: فحص الفروق المنتهية

كيف نعرف أنّ `backward` المكتوب يدويًّا *صحيح*؟ بمقارنته بتقديرٍ مبنيّ على تعريف
المشتقّة لا يحتاج إلى أيّ تفاضل:

```
numerical_grad ≈ ( f(x + ε) − f(x − ε) ) / 2ε
```

ادفع مُدخلًا واحدًا بمقدار `ε` ضئيل، وأعِد قياس المخرَج، ثمّ اقسِم. إنّه بطيء أكثر
ممّا ينبغي للتدريب (تمرير أمامي *لكلّ وزن*)، لكنّه مثاليّ للاختبار. وكلّ عملية قابلة
للاشتقاق في BabyTorch يُتحقَّق منها على هذا النحو مقابل التدرُّج التحليلي — انظر
`check_gradient` في
[`tests/conftest.py`](../../tests/conftest.py) والمجموعة في
[`tests/test_autograd.py`](../../tests/test_autograd.py). ومتى تطابق الاثنان إلى
خمس منازل عشرية على مُدخلات عشوائية، فالتفاضل صحيح.

## إطفاء المُسجِّل

تسجيل المخطّط يكلّف ذاكرة (فكلّ نتيجة وسيطة تبقى حيّة من أجل التمرير الخلفي). وأثناء
التقييم وتوليد النصّ لا يوجد تمرير خلفي، فنُطفئ التسجيل:

```python
with babytorch.no_grad():
    predictions = model(x)      # forward only, nothing remembered
```

**جرِّبه**

```python
>>> import babytorch
>>> w = babytorch.tensor([[1.0, -2.0], [3.0, 0.5]], requires_grad=True)
>>> loss = ((w @ w) ** 2).mean()
>>> loss.backward()
>>> w.grad                     # dLoss/dw, computed through @, **2 and mean
```

غيِّر التعبير إلى ما شئت — كلّ تركيبة من عمليات هذا الفصل تُشتَقّ تلقائيًّا. تلك
القابلية للتركيب هي مردود التصميم، وهي السبب الوحيد الذي يتيح للفصل 7 أن يبني GPT
دون كتابة سطر واحد من شيفرة التدرُّج.

## تمارين

**اختبر فهمك** (الإجابات تُفتَح):

**س1.** في `y = x * x`، يُغذّي المُوَتِّر `x` عمليةَ الضرب مرّتين. ماذا يفعل
`backward()` بإسهامَي التدرُّج، وأيّ سطر من شيفرة الاجتياز يقرّر ذلك؟

<details><summary>الإجابة</summary>

إنّه **يجمعهما** — `tensor.grad = tensor.grad + tensor_grad` في المسير المعكوس.
قاعدة السلسلة تجمع على كلّ المسارات من الخسارة إلى مُوَتِّر. (والتراكم نفسه عبر
*الدُّفعات* هو سبب استدعاء حلقات التدريب لـ `zero_grad()`.)

</details>

**س2.** الفروق المنتهية تحسب تدرُّجات صحيحة دون أيّ تفاضل البتّة. فلماذا لا نُدرِّب
بها بدلًا من الانتشار العكسي؟

<details><summary>الإجابة</summary>

الكلفة: تمريران أماميّان **لكلّ مُعامِل** في كلّ خطوة — فلنموذج BabyGPT ذي 2.7 مليون
مُعامِل، ملايين التمريرات الأمامية من أجل تحديث واحد. أمّا الانتشار العكسي فيُسلّم
كلّ تدرُّج في تمرير أماميّ واحد زائد تمرير خلفيّ واحد تقريبًا. الفروق المنتهية
لـ*اختبار* التفاضل، لا لإجرائه.

</details>

**س3.** يصل مُدخل إلى `relu` بقيمة سالبة. أيّ تدرُّج يعود عبره، وأيّ نمط إخفاق
يُنشئه ذلك؟

<details><summary>الإجابة</summary>

صفر تمامًا — فالميل المحلّي لـ `max(0, x)` تحت الصفر هو 0. والعصبون الذي يكون مُدخله
*دائمًا* سالبًا لا يتعلّم شيئًا بعد ذلك أبدًا («ReLU ميّت»). لهذا تهمّ مقاييس
التهيئة، ولهذا يوجد النوع المُتسرِّب (`alpha > 0`).

</details>

**طبِّقه بنفسك** — نفِّذ `MinOperation` و★ `AbsOperation` (التمرير الأمامي *و*
الخلفي) في
[`exercises/ch02_autograd.py`](../exercises/ch02_autograd.py)، ثمّ شغّل
`pytest book/exercises/test_ch02_autograd.py -v`. تفاضلك يواجه قاضي الفروق
المنتهية نفسه الذي تواجهه عمليات المكتبة ذاتها.
([كيف تعمل التمارين](../exercises/README.md).)

---

**ملفّات المصدر لهذا الفصل:**
[`babytorch/engine/operations.py`](../../babytorch/engine/operations.py) (العمليات) ·
[`babytorch/engine/tensor.py`](../../babytorch/engine/tensor.py) (`backward`، `no_grad`) ·
[`tests/test_autograd.py`](../../tests/test_autograd.py) (البرهان على أنّه يعمل)

[→ الفصل 1: المُوَتِّرات](01-tensors.md) | [المحتويات](README.md) | [الفصل 3: الشبكات العصبية ←](03-neural-networks.md)
