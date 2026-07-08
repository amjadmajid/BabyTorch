# الفصل 1 — المُوَتِّرات

*الجزء الأول، الفصل 1 من 4. كلّ شيء في التعلّم العميق — الصور، والجُمل، وأوزان
النماذج، والتدرُّجات — يُخزَّن في بنية بيانات واحدة. هذا الفصل يدور حول تلك
البنية.*

## فكرة واحدة: كتلة من الأرقام لها شكل

**المُوَتِّر** (tensor) مصفوفة متعدّدة الأبعاد: كتلة من الأرقام مع **شكل** (shape)
يصف كيف تُنظَّم تلك الكتلة.

![المُوَتِّرات كُتَل من الأرقام لها شكل — عدد قياسي، ومتّجه، ومصفوفة، ومُوَتِّر ثلاثي الأبعاد — والبثّ يمدّ مُوَتِّرًا بحجم 1 ليخدم دُفعة كاملة](../figures/fig-tensors.svg)

لماذا يُصرّ التعلّم العميق على هذه البنية بدلًا من قوائم بايثون؟ لسببين:

1. **المعنى يسكن في الشكل.** دُفعة من 32 جملة، طول كلٍّ منها 128 رمزًا، وكلّ رمز
   متّجه من 192 رقمًا، هي مُوَتِّر شكله `(32, 128, 192)`. الشكل *هو* سجلّ الحسابات.
2. **السرعة تسكن في الكتلة.** لأنّ الأرقام تقع في كتلة ذاكرة واحدة متّصلة، تُنفَّذ
   العمليات على المُوَتِّر كاملًا كحلقات مُصرَّفة محكمة (أو نوى GPU) بدلًا من بايثون
   المُفسَّر. القاعدة الذهبية للبرمجة بالمصفوفات: *لا تُكرِّر على العناصر في بايثون؛
   بل قُل ما تريد فعله بالكتلة كاملة.*

## إنشاء المُوَتِّرات

يُحاكي BabyTorch دوالّ الإنشاء في PyTorch
([`babytorch/__init__.py`](../../babytorch/__init__.py)):

```python
import babytorch

a = babytorch.tensor([[1., 2., 3.], [4., 5., 6.]])  # from data
z = babytorch.zeros(2, 3)         # all zeros
o = babytorch.ones(2, 3)          # all ones
r = babytorch.randn(2, 3)         # random, normal distribution (mean 0, std 1)
u = babytorch.rand(2, 3)          # random, uniform in [0, 1)
n = babytorch.arange(0, 10, 2)    # 0, 2, 4, 6, 8

babytorch.manual_seed(42)         # make the random ones reproducible
```

العناصر من نوع `float32` افتراضيًّا، تمامًا كما في PyTorch — نادرًا ما يحتاج
التعلّم العميق إلى دقّة أعلى، والأرقام الأصغر تعني حسابًا أسرع.

**جرِّبه**

```python
>>> import babytorch
>>> t = babytorch.tensor([[1., 2., 3.], [4., 5., 6.]])
>>> t.shape
(2, 3)
>>> t.ndim        # how many dimensions
2
>>> t.size        # how many numbers in total
6
>>> t.sum().item()   # .item() unwraps a single-element tensor to a Python number
21.0
```

## إعادة التشكيل: الأرقام نفسها، تنظيم مختلف

يمكن إعادة تنظيم كتلة الأرقام دون نسخ أيّ شيء:

```python
x = babytorch.arange(6)      # shape (6,):    [0 1 2 3 4 5]
x.reshape(2, 3)              # shape (2, 3):  [[0 1 2] [3 4 5]]
x.reshape(3, 2)              # shape (3, 2):  [[0 1] [2 3] [4 5]]

m = babytorch.randn(2, 3)
m.T                          # transpose: shape (3, 2), rows <-> columns
m.unsqueeze(0)               # insert a size-1 axis:  (2, 3) -> (1, 2, 3)
m.unsqueeze(0).squeeze()     # remove size-1 axes:    (1, 2, 3) -> (2, 3)
```

تبدو هذه شكليّة الآن، لكنّ الجزء الثاني يعتمد عليها باستمرار: الفصل 6 يُقسّم
مُوَتِّرًا شكله `(B, T, C)` إلى رؤوس انتباه بلا شيء سوى `reshape` و`transpose`.

## الحساب على المُوَتِّرات كاملة

تُطبَّق العمليات الحسابية **عنصرًا بعنصر**، زوجًا مُحاذًى في كلّ مرّة:

```python
a = babytorch.tensor([1., 2., 3.])
b = babytorch.tensor([10., 20., 30.])
(a + b).data      # [11. 22. 33.]
(a * b).data      # [10. 40. 90.]  (element-wise, NOT matrix product)
(a ** 2).data     # [1. 4. 9.]
```

أمّا المُعامِل `@` فهو **ضرب المصفوفات** الحقيقي — صفوف تُضرب نقطيًّا بأعمدة. وهو
العملية الأهمّ على الإطلاق في التعلّم العميق؛ يقضي نموذج GPT جُلّ وقته داخل `@`:

```
        (2, 3)      @      (3, 4)      ->      (2, 4)
     [[. . .]           [[. . . .]           [[. . . .]
      [. . .]]           [. . . .]            [. . . .]]
                         [. . . .]]
          └── inner dimensions must match ──┘
              (3 columns) dot (3 rows)
```

```python
a = babytorch.randn(2, 3)
w = babytorch.randn(3, 4)
(a @ w).shape     # (2, 4)
```

عمليات الاختزال تُقلِّص الأبعاد: `t.sum()` و`t.mean()` و`t.max()` و`t.var()` —
على كلّ شيء، أو على امتداد محور واحد (`t.sum(axis=0)` يجمع الصفوف فيُزيلها).

## البثّ: المُوَتِّرات الصغيرة تتمدّد لتُلائم

حين تختلف الأشكال، تقوم مكتبة المصفوفات **بالبثّ**: تنسخ (افتراضيًّا) الأبعاد ذات
الحجم 1 حتى تتطابق الأشكال.

```
      x: (32, 10)      +      b: (1, 10)

      [[..........]           [[----------]     <- the same row,
       [..........]    +       [----------]        virtually repeated
           ...                     ...             32 times
       [..........]]           [----------]]
```

```python
x = babytorch.randn(32, 10)   # a batch of 32 examples
b = babytorch.ones(1, 10)     # ONE bias row
y = x + b                     # b is stretched across all 32 rows
y.shape                       # (32, 10)
```

هكذا يخدم متّجه انحياز واحد دُفعة كاملة، وهكذا تضرب مصفوفة أوزان واحدة كلّ جملة
في الدُّفعة دفعةً واحدة. القاعدة، بمحاذاة الأشكال من اليمين: بُعدان متوافقان حين
يتساويان، أو حين يكون أحدهما 1 (والأبعاد البادئة المفقودة تُحتسب 1).

تذكّر البثّ — فهو يعود في الفصل 2 بمفاجأة: كلّ ما *نُسِخ* في التمرير الأمامي يجب
أن *يُجمَع* في التمرير الخلفي.

## الشيفرة نفسها على المعالج وعلى GPU

لا يذكر BabyTorch `numpy` مباشرةً أبدًا. كلّ وحدة تستورد مكتبة المصفوفات تحت الاسم
المحايد `xp` من [`babytorch/backend.py`](../../babytorch/backend.py):

```python
from babytorch.backend import xp    # numpy on CPU, cupy on GPU

xp.zeros((2, 3))                    # works identically on both
```

يختار `backend.py` المكتبة — **CuPy** إن توفّرت بطاقة GPU من NVIDIA، وإلّا
**NumPy** — و`xp` وسيط صغير يُمرِّر كلّ نداء إلى الخيار الحالي. وCuPy نسخة مطابقة
عمدًا لواجهة NumPy، فهذا الخيار الواحد هو قصّة GPU بأكملها — لا توجد أيّ شيفرة أخرى
خاصّة بالـ GPU في الإطار. تحكّم فيه من الشيفرة، أو من البيئة:

```python
>>> import babytorch
>>> babytorch.set_device("cpu")    # "cpu", "cuda", or "auto"
'cpu'
>>> babytorch.device()
'cpu'
>>> t.numpy()    # copy back to a NumPy array on the CPU (for plotting, saving...)
```

```bash
BABYTORCH_DEVICE=cpu  python train.py    # initial device via the environment
BABYTORCH_DEVICE=cuda python train.py    # require the GPU
```

قاعدة واحدة: اختر الجهاز *قبل* بناء المُوَتِّرات أو النماذج — فالمصفوفات لا تنتقل
إلى المكتبة الجديدة بعد إنشائها. (على macOS لا يوجد CUDA، فيعمل كلّ شيء على المعالج
— وهو ما يعمل مباشرةً دون عناء.)

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/backend.py</code></summary>

```python
class _XP:
    """Proxy that forwards every attribute to the active array library.

    Because modules bind ``xp`` once (``from .backend import xp``) but
    every *use* is an attribute access (``xp.zeros``), routing the
    lookup through ``__getattr__`` lets :func:`set_device` swap the
    library underneath all of them at once.  For MLX the "library" is the
    :mod:`babytorch.mlx_backend` adapter module rather than MLX itself.
    """

    _lib = None  # numpy, cupy, or the mlx_backend adapter; set by set_device()

    def __getattr__(self, name):
        return getattr(_XP._lib, name)
```

```python
def set_device(name):
    """Select the array library: ``"cpu"``, ``"cuda"`` (or ``"gpu"``),
    ``"mps"`` (or ``"mlx"``), or ``"auto"``.  Returns the name of the device
    actually selected.

    Call it *before* creating tensors or models (see the module docstring
    for why).  ``"cuda"`` and ``"mps"`` raise with an explanation if their
    GPU stack is not present; ``"auto"`` never raises.
    """
    global DEVICE
    name = name.lower()
    if name in ("cuda", "gpu"):
        _XP._lib = _cuda_library()
        DEVICE = "cuda"
    elif name in ("mps", "mlx"):
        _XP._lib = _mlx_library()
        DEVICE = "mps"
    elif name == "cpu":
        import numpy
        _XP._lib = numpy
        DEVICE = "cpu"
    elif name == "auto":
        try:
            return set_device("cuda")
        except Exception:
            return set_device("cpu")
    else:
        raise ValueError(
            f"unknown device {name!r}: expected 'cpu', 'cuda', 'mps' or 'auto'")
    return DEVICE
```

</details>

## مُوَتِّر يتذكّر من أين أتى

حتى الآن، لم يحتج شيء ممّا سبق إلى إطار عمل — فبايثون العادي (NumPy) يفعل كلّ ذلك.
وهذه هي البداية الفعلية لصنف `Tensor` في BabyTorch
([`babytorch/engine/tensor.py`](../../babytorch/engine/tensor.py)):

```python
self.data = xp.asarray(data, dtype=dtype)   # the block of numbers
self.requires_grad = requires_grad          # should gradients flow here?
self.grad = None                            # filled in by backward()
self.operation = None                       # the Operation that produced this
```

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/engine/tensor.py</code></summary>

```python
class Tensor:
    """A minimal tensor object that records operations for autodiff."""

    def __init__(self, data, requires_grad=False, dtype=None,
                 label="", _op_label=""):
    # ...
        # dtype=None means "keep the data's own dtype if it already has one
        # (so operations preserve float64/float32 through the graph), and
        # fall back to float32 for raw Python numbers/lists" -- matching
        # PyTorch, whose default tensor type is float32.
        if dtype is None:
            dtype = getattr(data, "dtype", None)
            if dtype is None or not xp.issubdtype(dtype, xp.floating):
                dtype = xp.float32
        # asarray: reuse the buffer when possible instead of always copying
        self.data = xp.asarray(data, dtype=dtype)
        self.dtype = self.data.dtype
        self.requires_grad = requires_grad
        self.grad = None
        self.operation = None
        self.label = label
        self._op_label = _op_label
    # ...
    @property
    def shape(self):
        return self.data.shape
    # ...
    def item(self):
        """Return the value of a single-element tensor as a Python number."""
        return self.data.item()

    def numpy(self):
        """Return the data as a NumPy array on the CPU (copies from GPU)."""
        return to_numpy(self.data)
```

</details>

ثلاثة حقول إضافية. أمّا `data` فقد غطّيناه؛ والحقول الثلاثة الأخرى موجودة لغرض
واحد: **كلّ مُوَتِّر يتذكّر العملية التي أنشأته، وتلك العملية تتذكّر مُوَتِّرات
دخلها.** اتبع تلك الروابط إلى الوراء تسترجع تاريخ الحساب بأكمله — كلّ `+` و`@`
و`reshape` قاد من المُدخلات إلى النتيجة.

يُسمّى ذلك التاريخ المُسجَّل *مخطّط الحساب*، وهو سرّ أُطُر التعلّم العميق كلّه:
لا تستطيع الشبكة أن *تتعلّم* إلّا إذا استطعنا حساب كيف تتغيّر الخسارة حين يتحرّك كلّ
وزن، والمخطّط يتيح للإطار أن يحسب ذلك تلقائيًّا — إلى الوراء، خطوةً مُسجَّلة في كلّ
مرّة.

وذلك هو موضوع الفصل 2.

## تمارين

**اختبر فهمك** (دقيقة لكلٍّ — الإجابات تُفتَح بالنقر):

**س1.** ما شكل `babytorch.randn(3, 1) * babytorch.randn(4)`؟

<details><summary>الإجابة</summary>

`(3, 4)`. بمحاذاة الأشكال من اليمين، يلتقي `(3, 1)` مع `(4,)` — والبُعد المفقود
يُحتسب 1، فيتمدّد كلاهما: عمود الـ 3 يتكرّر عبر 4 أعمدة، وصفّ الـ 4 يتكرّر عبر
3 صفوف.

</details>

**س2.** `(32, 10) + (10,)` يُبثّ بلا مشكلة، لكنّ `(32, 10) + (32,)` يرفع خطأً.
لماذا؟

<details><summary>الإجابة</summary>

المحاذاة من **اليمين**: يصطفّ `(32,)` مقابل البُعد الأخير ذي الحجم 10، و32 ≠ 10
دون أن يكون أيٌّ منهما 1. (ولإضافة رقم واحد لكلّ *صفّ* تُعيد التشكيل إلى `(32, 1)`.)

</details>

**س3.** لماذا `float32` هو النوع الافتراضي بدل `float64` الأدقّ؟

<details><summary>الإجابة</summary>

نصف الذاكرة وضِعف الإنتاجية، والانحدار التدرُّجي مُشوَّش أصلًا — ضوضاء الدُّفعة تُقزِّم
خطأ التقريب الإضافي، فلن تشتري الدقّة الأعلى شيئًا. (نماذج الطليعة تذهب إلى دقّة
*أقلّ*: 16 بتًّا فما دون.)

</details>

**طبِّقه بنفسك** — المسار الأعمق، المُقيَّم بالاختبارات: نفِّذ `standardize`
و★ `outer` (بالبثّ فقط، دون `@`) في
[`exercises/ch01_tensors.py`](../exercises/ch01_tensors.py)، ثمّ شغّل
`pytest book/exercises/test_ch01_tensors.py -v`.
([كيف تعمل التمارين](../exercises/README.md).)

---

**ملفّات المصدر لهذا الفصل:**
[`babytorch/__init__.py`](../../babytorch/__init__.py) (دوالّ الإنشاء) ·
[`babytorch/backend.py`](../../babytorch/backend.py) (اختيار المعالج/الـ GPU) ·
[`babytorch/engine/tensor.py`](../../babytorch/engine/tensor.py) (صنف Tensor)

[المحتويات](README.md) | [الفصل 2: الاشتقاق التلقائي ←](02-autograd.md)
