# الفصل 4 — التدريب

*الجزء الأول، الفصل 4 من 4. لدينا نماذج تحسب، واشتقاق تلقائي (autograd)
يشتقّها. هذا الفصل يُغلِق الحلقة — حرفيًّا: الحلقة ذات الخطوات الأربع التي تُحوِّل
الأوزان العشوائية إلى شبكة مُدرَّبة، والمُحسِّنات (optimizers) التي تجعل ذلك سريعًا.*

## الانحدار التدرُّجي في صورة واحدة

بعد استدعاء `loss.backward()`، يحمل كلّ مُعامِل قيمة `dLoss/dparameter` — الاتّجاه
الذي *تزداد* فيه الخسارة. لذا اخطُ في الاتّجاه المعاكس:

```
parameter  =  parameter  −  learning_rate × gradient
```

افعل ذلك لكلّ مُعامِل، وأعِد قياس الخسارة على بيانات جديدة، ثمّ كرِّر. تخيّل نفسك
واقفًا على منحدرٍ يلفّه الضباب حيث الخسارة هي ارتفاعك: التدرُّج هو الميل تحت قدميك،
وكلّ تحديث خطوة صغيرة نزولًا. لا تستطيع رؤية الوادي — لكنّك لست بحاجة إلى ذلك. كلّ
جزء من تدريب التعلّم العميق، وصولًا إلى نماذج GPT بما فيها، هو هذه الحلقة:

```python
for step in range(num_steps):
    predictions = model(x)               # 1. forward
    loss = criterion(predictions, y)     # 2. how wrong?
    optimizer.zero_grad()                #    (forget old gradients)
    loss.backward()                      # 3. gradients for every parameter
    optimizer.step()                     # 4. small step downhill
```

![حلقة التدريب ذات الخطوات الأربع: التمرير الأمامي يحسب التنبّؤات، والخسارة تقول مدى خطئها، والتمرير الخلفي يُنتج تدرُّجًا لكلّ مُعامِل، والمُحسِّن يخطو نزولًا — مع zero_grad الذي يمسح التدرُّجات القديمة قبل كلّ دورة](../figures/fig-training-loop.svg)

لماذا `zero_grad()` في كلّ مرّة؟ من الفصل 2: `backward()` **يُراكِم** في `.grad`
(ولا بدّ له من ذلك — فالمُعامِل المُستخدَم مرّتين يجمع إسهاماته). دون إعادة التصفير،
سترث الدُّفعة الثانية تدرُّجات الدُّفعة الأولى. ونسيان هذا السطر هو خطأ المبتدئين
الكلاسيكي: يتعثّر التدريب، ولا ينهار شيء.

## مثال كامل يمكنك تشغيله

لِنُلائم خطًّا مستقيمًا لبيانات مشوَّشة — أصغر «نموذج» ممكن:

```python
import babytorch
import babytorch.nn as nn
from babytorch.optim import SGD

babytorch.manual_seed(0)

# Data from a secret rule: y = 3x + 2, plus noise
x = babytorch.rand(100, 1) * 4.0 - 2.0
y = x * 3.0 + 2.0 + babytorch.randn(100, 1) * 0.3

model = nn.Linear(1, 1)                  # w and b start ~random
optimizer = SGD(model.parameters(), learning_rate=0.1)
criterion = nn.MSELoss()

for step in range(200):
    loss = criterion(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

print("w =", model.w.item(), " b =", model.b.item())   # ≈ 3 and ≈ 2
```

يستعيد النموذج القاعدة السرّية من الأمثلة وحدها. وكلّ درس تعليمي في هذا المستودع —
حتى BabyGPT نفسه — هو هذا البرنامج ذاته مع `model` أكبر و`(x, y)` أكثر إثارة.

## معدّل التعلّم: المقبض الوحيد الذي يجب أن تحترمه

يتحكّم `learning_rate=0.1` في حجم الخطوة، وهو أكثر رقم مصيري في التدريب:

```
too small ────────── just right ────────── too large
loss creeps down     loss falls fast,      loss oscillates,
imperceptibly        then settles          or explodes to NaN
```

لا توجد قيمة كونيّة واحدة — `0.1` تناسب انحدارًا صغيرًا، و`3e-3` تناسب BabyGPT،
والمُحوِّلات الكبيرة (Transformers) تستخدم معدّلات أصغر من ذلك. وحين يسيء التدريب
التصرّف، اشتبه في معدّل التعلّم أوّلًا.

## الدُّفعات الصغيرة

لا تتّسع مجموعات البيانات الحقيقية في تمرير أمامي واحد. بدلًا من ذلك، تستخدم كلّ
خطوة **دُفعة صغيرة** (mini-batch) عشوائية (32 مثالًا، مثلًا). ويصبح التدرُّج تقديرًا
لتدرُّج مجموعة البيانات كاملة — مشوَّشًا، لكنّه أرخص بكثير حتى إنّ أخذ خطوات مشوَّشة
كثيرة يتفوّق على أخذ خطوات قليلة مثالية (بل إنّ الضوضاء تساعد على الهروب من المناطق
السيّئة). هذا هو معنى «العشوائي» (stochastic) في SGD. ويتولّى
[`DataLoader`](../../babytorch/datasets/data_loader.py) الخلط والتقطيع:

```python
from babytorch.datasets import DataLoader
for x_batch, y_batch in DataLoader(dataset, batch_size=32, shuffle=True):
    ...one training step...
```

## خطوات أذكى: عائلة المُحسِّنات

تتشارك كلّ المُحسِّنات واجهةً واحدة — `step()` و`zero_grad()` — وتقيم في
[`babytorch/optim/optim.py`](../../babytorch/optim/optim.py). ثلاثة أجيال، كلٌّ
منها يُصلِح مشكلة حقيقية:

**SGD** — القاعدة البسيطة أعلاه. له خياران مفيدان: *الزخم* (momentum) يحتفظ بسرعة
متجمّعة (`v = momentum·v + grad`) بحيث تتراكم السرعة في الاتّجاهات الثابتة وتتلاشى
ضوضاء التعرُّج، كأنّها كرة ثقيلة تتدحرج نزولًا؛ و*اضمحلال الأوزان* (weight_decay)
يُقلِّص الأوزان باستمرار نحو الصفر (تنظيم L2) لتثبيط القيم المتطرّفة.

**Adam** — نسخة من SGD حيث يضبط كلّ مُعامِل حجم خطوته *الخاصّة*. يتتبّع مُتوسِّطين
متجمّعين لكلّ مُعامِل: `m`، مُتوسِّط التدرُّج (أيّ اتّجاه هو نزولًا؟)، و`v`، مُتوسِّط
*مربّع* التدرُّج (ما حجم الخطوات هنا وكم هي مشوَّشة؟)، ثمّ يُحدِّث

```
p  =  p − learning_rate · m / (√v + ε)
```

المُعامِلات ذات التدرُّجات الضخمة على الدوام تحصل على خطوات أصغر نسبيًّا؛ والنادرة
لكن المهمّة تحصل على خطوات أكبر نسبيًّا. (تقسم الشيفرة أيضًا على `1 − βᵗ` — *تصحيح
الانحياز* (bias correction)، الذي يُزيل انحياز المُتوسِّطات التي تبدأ من الصفر.)
وAdam هو الخيار الافتراضي المعقول لأيّ شيء أكبر من مجرّد لعبة.

**AdamW** — نسخة Adam مع *فصل* اضمحلال الأوزان: يُطبَّق التقليص مباشرةً على الأوزان
بدلًا من خلطه في التدرُّج (حيث يشوّهه تحجيم Adam الخاصّ بكلّ مُعامِل). يُنظِّم بصورة
أفضل عمليًّا — وهذا ما تتدرّب به نماذج GPT، بما فيها نموذجنا في الفصل 8.

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/optim/optim.py</code> (خطوة SGD، ثمّ خطوة AdamW)</summary>

```python
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            if self.momentum > 0:
                if self.velocities[i] is None:
                    self.velocities[i] = xp.zeros_like(p.data)
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                grad = self.velocities[i]
            p.data -= self.learning_rate * grad
    # ...
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Decoupled decay: shrink the weight directly...
            if self.weight_decay > 0:
                p.data -= self.learning_rate * self.weight_decay * p.data
            # ...then take the adaptive step.
            p.data -= self.learning_rate * m_hat / (xp.sqrt(v_hat) + self.eps)
```

</details>

## جداول معدّل التعلّم

المعدّل الصحيح عند الخطوة 0 يكون عادةً أكبر من اللازم قرب النهاية — ففي البداية
تريد خطوات جريئة عبر المشهد، ولاحقًا تريد استقرارًا حذرًا في الوادي. **المُجدوِلات**
(schedulers)
([`babytorch/optim/lr_scheduler.py`](../../babytorch/optim/lr_scheduler.py))
تضبط `optimizer.learning_rate` عبر الزمن؛ استدعِ `scheduler.step(t)` في كلّ تكرار
أو حقبة. يُغطّي `StepLR` (يهبط 10× كلّ N حقبة) و`LambdaLR` (أيّ دالّة تريدها)
التدريب الكلاسيكي؛ أمّا ما تستخدمه نماذج GPT فهو **`CosineWarmupLR`**:

```
lr │        ╭──╮
   │      ╱      ╲
   │    ╱           ╲
   │  ╱                ╲──
   │╱                       ╲────────
   └────────────────────────────────► step
    warmup:        cosine decay:
    ramp 0 → lr    smooth glide to min_lr
```

الإحماء مهمّ لأنّ مُتوسِّطات Adam المتجمّعة تكون عديمة القيمة في العشرات الأولى من
الخطوات — خطوات صغيرة حتى تستقرّ إحصاءاته، ثمّ السرعة القصوى، ثمّ هبوط جيبي (cosine)
سلس.

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>babytorch/optim/lr_scheduler.py</code></summary>

```python
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        super().__init__(optimizer)
        assert 0 <= warmup_steps < total_steps
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def step(self, t):
        if t < self.warmup_steps:
            lr = self.base_lr * (t + 1) / self.warmup_steps
        elif t >= self.total_steps:
            lr = self.min_lr
        else:
            progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine
        self.optimizer.learning_rate = lr
```

</details>

## هل تعلّم فعلًا؟ التدريب مقابل التحقّق

يستطيع النموذج أن يتفوّق في الأسئلة التي تدرّب عليها عن طريق *حفظها* — خسارة تبدو
مبهرة، ونموذج عديم الفائدة. لذا نحتجز دائمًا بعض البيانات:

```
train loss ↓, val loss ↓        learning         keep going
train loss ↓, val loss ↑        overfitting      stop / regularize
train loss stuck high           underfitting     bigger model, higher lr,
                                                 or a bug (check zero_grad!)
```

الفجوة بين المنحنيَين هي صدق النموذج. وعلاجات الإفراط في المُلاءمة (overfitting)،
بترتيب تجربتها: مزيد من البيانات، ثمّ `Dropout`، ثمّ `weight_decay`، ثمّ نموذج
أصغر. سترى هذا التقسيم مُستخدَمًا في الفصل 8، حيث يحتجز BabyGPT 10% من نصوص شكسبير
ويطبع الخسارتين كلّ 100 خطوة.

لمراقبة عمليات التشغيل، يرسم `babytorch.Grapher`
([`babytorch/visualization/grapher.py`](../../babytorch/visualization/grapher.py))
منحنيات الخسارة (`plot_loss`) — والأغرب من ذلك أنّه يرسم مخطّط الحساب الفعلي لأيّ
مُوَتِّر (`show_graph`)، راسمًا مخطّطات الفصل 2 فعليًّا من روابط `operation`
المُسجَّلة.

**جرِّبه** — الاختبار غير الخطّي الكلاسيكي، XOR. لا يمكن لطبقة خطّية واحدة أن
تحلّه، وهذا مُبرهَن؛ أمّا مع طبقة خفيّة واحدة وانحناءة، فيجد الانحدار التدرُّجي حلًّا
في ثوانٍ:

```python
import babytorch, babytorch.nn as nn
from babytorch.optim import SGD

babytorch.manual_seed(1)
X = babytorch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = babytorch.tensor([[0.],[1.],[1.],[0.]])         # XOR truth table

model = nn.Sequential(nn.Linear(2, 8, nn.Tanh()),
                      nn.Linear(8, 1, nn.Sigmoid()))
optimizer = SGD(model.parameters(), learning_rate=0.5)
criterion = nn.MSELoss()

for step in range(2000):
    loss = criterion(model(X), Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(X).data.round().T)    # [[0. 1. 1. 0.]] — XOR, learned
```

للاطّلاع على أمثلة أكبر مُفصَّلة، انظر
[`tutorials/regression`](../../tutorials/regression) و
[`tutorials/classification`](../../tutorials/classification) (وصولًا إلى أرقام
MNIST بالالتفاف)، و[`tests/test_training.py`](../../tests/test_training.py) —
البراهين الشاملة على أنّ هذه الحلقة تعمل، بما في ذلك نموذج GPT مُصغَّر.

## تمارين

**اختبر فهمك** (الإجابات تُفتَح بالنقر):

**س1.** اكتشف الخطأ:

```python
for step in range(1000):
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

<details><summary>الإجابة</summary>

لا يوجد `zero_grad()`. كلّ `backward()` **يضيف** إلى `.grad` الموجود، فبحلول
الخطوة 100 يخطو النموذج على امتداد مجموع 100 تدرُّج قديم. ولا ينهار شيء — بل تتوقّف
الخسارة أو تتذبذب فحسب، وهذا ما يجعله الخطأ الصامت الكلاسيكي.

</details>

**س2.** تهبط الخسارة هبوطًا جميلًا على مدى 300 خطوة، ثمّ تقرأ فجأةً `nan`. أيّ مقبض
تشتبه فيه أوّلًا، وما حزام الأمان المعياري؟

<details><summary>الإجابة</summary>

معدّل التعلّم — فخطوة أكبر من اللازم تجاوزت إلى منطقة ذات تدرُّجات ضخمة فانفجرت
التحديثات. اخفِضه؛ وقُصّ معيار التدرُّج (ستبني `clip_grad_norm_` في تمارين هذا
الفصل) حتى لا تستطيع دُفعة واحدة سيّئة الحظّ أن تقذف الأوزان بعيدًا.

</details>

**س3.** خسارة التدريب تواصل الهبوط؛ وخسارة التحقّق انعطفت صعودًا. سمِّ الحالة
وعلاجَين.

<details><summary>الإجابة</summary>

الإفراط في المُلاءمة — النموذج يحفظ مجموعة التدريب. العلاجات، بترتيب التجربة: مزيد
من البيانات، ثمّ `Dropout` / `weight_decay`، ثمّ نموذج أصغر، أو ببساطة التوقّف عند
الحدّ الأدنى لخسارة التحقّق (الإيقاف المبكر).

</details>

**طبِّقه بنفسك** — نفِّذ `clip_grad_norm_` و★ مُحسِّن `RMSProp` كاملًا (الحلقة
المفقودة بين SGD وAdam) في
[`exercises/ch04_training.py`](../exercises/ch04_training.py)، ثمّ شغّل
`pytest book/exercises/test_ch04_training.py -v`.
([كيف تعمل التمارين](../exercises/README.md).)

---

**اكتمل الجزء الأول.** تعرف الآن الآلة بأكملها: المُوَتِّرات تحمل البيانات (الفصل
1)، والاشتقاق التلقائي يشتقّ أيّ شيء مبنيّ منها (الفصل 2)، والوحدات تُنظِّم
المُعامِلات في نماذج (الفصل 3)، وحلقة التدريب تُحسِّنها (الفصل 4). لا يضيف الجزء
الثاني أيّ آليّة جديدة — فنموذج GPT *مجرّد `Module` آخر* تُدرِّبه *هذه الحلقة نفسها
تمامًا*. ما يضيفه الجزء الثاني هو الأفكار: كيف يصير النصّ مُوَتِّرات، والبنية التي
حوّلت التنبّؤ بالرمز التالي إلى التقنية الكامنة وراء ChatGPT.

**ملفّات المصدر لهذا الفصل:**
[`babytorch/optim/optim.py`](../../babytorch/optim/optim.py) ·
[`babytorch/optim/lr_scheduler.py`](../../babytorch/optim/lr_scheduler.py) ·
[`babytorch/datasets/data_loader.py`](../../babytorch/datasets/data_loader.py) ·
[`babytorch/visualization/grapher.py`](../../babytorch/visualization/grapher.py)

[→ الفصل 3: الشبكات العصبية](03-neural-networks.md) | [المحتويات](README.md) | [الجزء الثاني — الفصل 5: التجزئة إلى رموز ←](05-tokenization.md)
