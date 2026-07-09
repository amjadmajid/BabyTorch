# الفصل 2 — الاشتقاق التلقائي

*الجزء الأول، الفصل 2 من 4. هذا هو قلب أي إطار عمل للتعلّم العميق. بعد هذا الفصل،
لن يبدو `loss.backward()` كأنه سحر. سترى أنه ببساطة: نسجل ما حدث في التمرير الأمامي،
ثم نشغل هذا التسجيل بالعكس لحساب التدرجات.*

## لماذا نحتاج التدرجات؟

الشبكة العصبية دالة فيها أرقام كثيرة قابلة للتعديل، نسميها **الأوزان**. أثناء التدريب
نريد تقليل رقم واحد اسمه **الخسارة** (loss): أي مقدار خطأ النموذج.

السؤال المهم لكل وزن هو:

> إذا غيّرت هذا الوزن قليلًا، هل ستزيد الخسارة أم تنقص؟ وبأي مقدار؟

إجابة هذا السؤال هي **المشتقة** أو **التدرج** (gradient): `dLoss/dweight`.

إذا عرفنا تدرج كل وزن، يمكننا تحريك الأوزان خطوة صغيرة في الاتجاه الذي يقلل الخسارة.
هذه هي فكرة التدريب الأساسية، وسنستخدمها عمليًا في الفصل 4. الجزء الصعب ليس الفكرة،
بل حساب ملايين التدرجات بسرعة وبدقة. هذه هي مهمة الاشتقاق التلقائي (autograd).

يعتمد autograd على شيئين: قاعدة رياضية واحدة، وبنية بيانات واحدة.

## القاعدة الرياضية: قاعدة السلسلة

الحسابات الكبيرة تتكوّن من خطوات صغيرة. إذا كان:

```text
y = f(g(x))
```

فإن:

```text
dy/dx  =  dy/dg  ·  dg/dx
```

أي أن معدلات التغير تتضاعف على طول السلسلة.

لذلك لا نحتاج إلى اشتقاق النموذج كله دفعة واحدة. يكفي أن تعرف كل عملية صغيرة مشتقتها
المحلية: `+` و`*` و`matmul` و`exp` وغيرها. بعد ذلك نربط هذه المشتقات معًا من الخسارة
رجوعًا إلى الأوزان. لا توجد خطوة واحدة صعبة بحد ذاتها؛ المشكلة فقط أن عدد الخطوات كبير.

## بنية البيانات: مخطط الحساب

في نهاية الفصل 1 رأينا أن كل تنسور في BabyTorch يحمل معلومات إضافية. الآن نرى لماذا.
جرّب هذا المثال:

```python
import babytorch
x = babytorch.tensor([2.0], requires_grad=True)
y = x * x + 3.0 * x            # y = x² + 3x  ->  y = 10
```

لم يحسب BabyTorch الرقم `10` فقط. كل عملية أنشأت الناتج عبر `Tensor._make_output`،
وهذه الدالة خزنت رابطًا إلى العملية. والعملية نفسها احتفظت بمراجع إلى مدخلاتها.
النتيجة هي مخطط حساب:

![مخطط الحساب للدالة y = x² + 3x عند x = 2: الأسهم المتصلة تحسب القيم وتسجلها في التمرير الأمامي؛ والأسهم الحمراء المتقطعة تعيد تشغيل التسجيل بالعكس، فتضرب المشتقات المحلية. وحين يصل مساران إلى x، نجمع التدرجات لنحصل على 7](../figures/fig-autograd.svg)

كل صندوق في هذا المخطط يعرف مشتقته المحلية. لحساب `dy/dx`، نبدأ من `y` ونسير إلى
الخلف. في كل خطوة نطبق قاعدة السلسلة. وإذا وصل التدرج إلى التنسور نفسه من أكثر من
مسار، نجمع الإسهامات:

```python
y.backward()
x.grad          # [7.]   because dy/dx = 2x + 3 = 7 at x = 2
```

بقية الفصل تجيب عن سؤالين: ماذا يوجد داخل كل صندوق؟ وكيف يمشي `backward()` في المخطط؟

## داخل الصندوق: عقد `Operation`

الرياضيات الفعلية موجودة في
[`babytorch/engine/operations.py`](../../babytorch/engine/operations.py).
كل عملية تنفذ دالتين:

* `forward` — يحسب الناتج من المدخلات، ويتذكر ما سيحتاجه لاحقًا في التمرير الخلفي.
* `backward(grad)` — يستقبل `dLoss/dOutput` ويعيد `dLoss/dInput` لكل مدخل.
  هذه خطوة واحدة فقط من قاعدة السلسلة.

هذا هو الضرب كاملًا:

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

اقرأ `backward` ببطء: حساسية الخسارة تجاه `a` هي التدرج القادم `grad` مضروبًا في
المشتقة المحلية للعملية بالنسبة إلى `a`، وهي `b`. هذا هو تطبيق قاعدة السلسلة في الكود.

كل عملية أخرى في الملف لها الشكل نفسه. الاختلاف فقط في المشتقة المحلية:

| العملية | القاعدة المحلية في `backward` |
|-----------|--------------------------|
| `a + b` | التدرج يمر كما هو إلى الطرفين |
| `a * b` | `a` يأخذ `grad·b`، و`b` يأخذ `grad·a` |
| `a ** n` | قاعدة القوة: `grad · n·aⁿ⁻¹` |
| `exp(a)` | `grad · exp(a)`، ويمكن استخدام الناتج المحفوظ |
| `relu(a)` | مرر التدرج حيث `a > 0`، واجعله صفرًا في غير ذلك |
| `sum` | انسخ التدرج راجعًا إلى كل عنصر شارك في الجمع |
| `max` | وجّه التدرج إلى العنصر الأكبر؛ وإذا تعادل أكثر من عنصر، قسّم التدرج بينهم |
| `a[idx]` | انثر التدرج راجعًا إلى المواضع التي قرأنا منها |

والقاعدة التي تستحق الحفظ، لأن Transformers تعتمد عليها كثيرًا، هي ضرب المصفوفات
`C = A @ B`:

```text
dL/dA = dL/dC @ Bᵀ           dL/dB = Aᵀ @ dL/dC
```

راجع الأشكال وسترى أنها لا تركب إلا بهذه الطريقة.

## فصل المسؤوليات

لاحظ ما لا يعرفه `MulOperation`: لا يعرف شيئًا عن مخطط الحساب، ولا عن
`requires_grad`، ولا عن طريقة سير `backward()`. العمليات في `operations.py` تهتم
بالرياضيات على المصفوفات الخام فقط.

أما تتبع المخطط فيوجد في [`babytorch/engine/tensor.py`](../../babytorch/engine/tensor.py).
كل دالة في `Tensor` تعمل كطبقة وصل صغيرة:

```python
def __mul__(self, other):
    op = MulOperation()
    result = op.forward(self, other)
    return self._make_output(
        op, result, self.requires_grad or other.requires_grad, "*"
    )
```

هذه نقطة تصميم مهمة: **الرياضيات في `operations.py`، وتتبع الحساب في `tensor.py`.**
عندما تفهم هذا الفصل بين المسؤوليات، يصبح المحرك كله أسهل بكثير.

## كيف يعمل التمرير الخلفي؟

`Tensor.backward()` في
[`tensor.py`](../../babytorch/engine/tensor.py) يفعل ثلاث خطوات:

**1. البذرة.** تدرج الخسارة بالنسبة إلى نفسها يساوي 1:

```python
self.grad = ones_like(self.data)
```

**2. ترتيب المخطط طوبولوجيًا.** نريد قائمة بكل التنسورات، بحيث يأتي كل تنسور بعد
التنسورات التي يعتمد عليها. يمكن عمل ذلك ببحث بسيط من الخسارة إلى الخلف:

```python
def build_topo(v):
    if v not in visited:
        visited.add(v)
        if v.operation:
            for tensor in v.operation.inputs():
                build_topo(tensor)
        topo.append(v)
```

**3. تشغيل القائمة بالعكس.** نمشي من الخسارة نحو المدخلات. كل عملية تأخذ تدرج ناتجها
وتحوله إلى تدرجات مدخلاتها:

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

الترتيب العكسي يضمن أنه عندما تصل إلى عملية ما، يكون تدرج ناتجها قد اكتمل بالفعل.
وهذه هي الخوارزمية كلها:

**الانتشار العكسي = ترتيب طوبولوجي + قاعدة السلسلة.**

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/engine/tensor.py</code> (كامل <code>backward()</code>، دون اختصار)</summary>

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

**لماذا نجمع التدرجات ولا نستبدلها؟** إذا استُخدم التنسور في أكثر من مكان، سيصل إليه
تدرج من كل استخدام. وقاعدة السلسلة تقول إن الإسهامات القادمة من المسارات المختلفة
تُجمع.

في مثالنا، `x` يدخل في `x*x` ويدخل أيضًا في `3*x`. كلا المسارين يؤثر في `y`، لذلك
كلاهما يجب أن يساهم في `x.grad`.

هذا هو السبب نفسه الذي يجعل حلقة التدريب تستدعي `zero_grad()` بين الدفعات. إذا لم
نصفّر التدرجات، ستبقى تدرجات الدفعة السابقة وتضاف إلى الجديدة. سنرى ذلك في الفصل 4.

## دين البث: `sum_to_shape`

في الفصل 1 رأينا أن البث يجعل تنسورًا صغيرًا يخدم تنسورًا أكبر. مثلًا، انحياز شكله
`(1, 10)` يمكن أن يضاف إلى دفعة شكلها `(32, 10)`.

في التمرير الأمامي، كأن صف الانحياز تكرر 32 مرة. في التمرير الخلفي يحدث العكس:
كل صف من الصفوف الـ 32 يرسل تدرجًا إلى الانحياز، لذلك يجب جمع هذه التدرجات كلها
والعودة إلى الشكل الأصلي `(1, 10)`.

هذه هي مهمة `Operation.sum_to_shape`، وقد رأيتها في `MulOperation.backward` أعلاه.
القاعدة الذهبية:

```text
broadcast (copy) in the forward pass   <=>   sum in the backward pass
```

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/engine/operations.py</code></summary>

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

## ثق، لكن تحقق: فحص الفروق المنتهية

كيف نعرف أن `backward` الذي كتبناه صحيح؟ نقارنه بطريقة عددية تعتمد على تعريف المشتقة
نفسه، ولا تحتاج إلى اشتقاق يدوي:

```text
numerical_grad ≈ ( f(x + ε) − f(x − ε) ) / 2ε
```

نغيّر عنصرًا واحدًا في المدخل بمقدار صغير جدًا `ε`، نقيس الناتج، ثم نغيره في الاتجاه
العكسي ونقيس مرة أخرى. الفرق بين القياسين يعطينا تقديرًا عدديًا للتدرج.

هذه الطريقة بطيئة جدًا للتدريب؛ لأنها تحتاج تمريرين أماميين لكل معامل. لكنها ممتازة
للاختبار. لذلك يختبر BabyTorch كل عملية قابلة للاشتقاق بمقارنتها مع هذا التقدير العددي.
انظر `check_gradient` في
[`tests/conftest.py`](../../tests/conftest.py) والاختبارات في
[`tests/test_autograd.py`](../../tests/test_autograd.py). إذا كان الفرق أقل من `1e-4`
على مدخلات عشوائية، فهذا دليل قوي أن التمرير الخلفي صحيح.

## إطفاء التسجيل

تسجيل مخطط الحساب يستهلك ذاكرة، لأن كل نتيجة وسيطة يجب أن تبقى محفوظة حتى نستخدمها
في التمرير الخلفي. لكن أثناء التقييم أو توليد النص، لا نحتاج إلى تدرجات. لذلك نطفئ
التسجيل:

```python
with babytorch.no_grad():
    predictions = model(x)      # forward only, nothing remembered
```

**جرّبه**

```python
>>> import babytorch
>>> w = babytorch.tensor([[1.0, -2.0], [3.0, 0.5]], requires_grad=True)
>>> loss = ((w @ w) ** 2).mean()
>>> loss.backward()
>>> w.grad                     # dLoss/dw, computed through @, **2 and mean
```

غيّر التعبير كما تريد. أي تركيبة من العمليات التي يعرفها BabyTorch يمكن اشتقاقها
تلقائيًا. هذه القابلية للتركيب هي قوة التصميم: في الفصل 7 سنبني GPT كاملًا دون كتابة
كود خاص لحساب تدرجاته.

## تمارين

**اختبر فهمك** (الإجابات تظهر عند الفتح):

**س1.** في `y = x * x`، يدخل التنسور `x` في عملية الضرب مرتين. ماذا يفعل
`backward()` بالتدرجين القادمين إلى `x`؟ وأي سطر في الكود يقرر ذلك؟

<details><summary>الإجابة</summary>

يجمعهما. السطر هو:

```python
tensor.grad = tensor.grad + tensor_grad
```

قاعدة السلسلة تجمع الإسهامات القادمة من كل المسارات التي تصل من الخسارة إلى التنسور.
والفكرة نفسها هي سبب استدعاء `zero_grad()` بين دفعات التدريب.

</details>

**س2.** الفروق المنتهية تحسب تدرجات صحيحة دون اشتقاق يدوي. فلماذا لا نستخدمها للتدريب
بدل الانتشار العكسي؟

<details><summary>الإجابة</summary>

لأن كلفتها ضخمة: تحتاج تمريرين أماميين لكل معامل في كل خطوة. إذا كان نموذج BabyGPT
فيه 2.7 مليون معامل، فهذا يعني ملايين التمريرات الأمامية من أجل تحديث واحد فقط.
أما الانتشار العكسي فيحسب كل التدرجات تقريبًا بكلفة تمرير أمامي واحد وتمرير خلفي واحد.
الفروق المنتهية ممتازة للاختبار، لا للتدريب.

</details>

**س3.** إذا دخلت قيمة سالبة إلى `relu`، ما التدرج الذي يعود عبرها؟ وما المشكلة التي
قد يسببها ذلك؟

<details><summary>الإجابة</summary>

يعود تدرج صفر؛ لأن ميل `max(0, x)` تحت الصفر يساوي 0. إذا ظل دخل عصبون ما سالبًا
دائمًا، فلن يتعلم هذا العصبون شيئًا، وهذا ما يسمى أحيانًا “ReLU ميت”. لذلك تهم
طريقة تهيئة الأوزان، ولهذا توجد نسخة Leaky ReLU حيث يكون `alpha > 0`.

</details>

**طبّقه بنفسك** — نفّذ `MinOperation` و★ `AbsOperation` (التمرير الأمامي والخلفي)
في [`exercises/ch02_autograd.py`](../exercises/ch02_autograd.py)، ثم شغّل
`pytest book/exercises/test_ch02_autograd.py -v`. سيحكم اختبار الفروق المنتهية على
حساباتك، كما يحكم على عمليات BabyTorch نفسها.
([كيف تعمل التمارين](../exercises/README.md).)

---

**ملفات المصدر لهذا الفصل:**
[`babytorch/engine/operations.py`](../../babytorch/engine/operations.py) (العمليات) ·
[`babytorch/engine/tensor.py`](../../babytorch/engine/tensor.py) (`backward`، `no_grad`) ·
[`tests/test_autograd.py`](../../tests/test_autograd.py) (البرهان على أنه يعمل)

[→ الفصل 1: التنسورات](01-tensors.md) | [المحتويات](README.md) | [الفصل 3: الشبكات العصبية ←](03-neural-networks.md)
