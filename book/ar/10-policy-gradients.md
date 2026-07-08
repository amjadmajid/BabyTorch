# الفصل 10 — تدرُّجات السياسة

حلّ الفصل 9 لعبة GridWorld بـ**جدول**: رقم واحد لكلّ حالة (state)، تملؤه
معادلة بلمان. ينجح ذلك لأنّ GridWorld فيها بضع عشرات من الحالات يمكن سردها.
لكنّه يصبح ميؤوسًا منه لحظة أن تصير الحالة صورةً أو رقعةً أو فقرةً — إذ تصير
الحالات كثيرة إلى حدٍّ فلكيّ، فلا جدول يسعها، ناهيك عن زيارة كلٍّ منها بما
يكفي لتعلُّمها.

تقوم بقيّة الجزء الثالث على فكرة واحدة تُفلت من الجدول: **استبدله بشبكة**.
تقرأ الشبكة الحالة و*تُعمِّم* — فتُعطي جوابًا معقولًا لحالات لم ترها قطّ. هذه
النقلة الواحدة تحوّل التعلّم المُعزَّز (reinforcement learning) الجدولي إلى
تعلّم مُعزَّز *عميق*، وتفتح طريقين. يسلك هذا الفصل الأول منهما: بدل تعلُّم
*قيمة* كلّ حالة، نتعلّم **كيف نتصرّف** مباشرةً — شبكة تقرأ الحالة وتُخرج فعلًا
(action). تلك الشبكة هي **السياسة** (policy)، وتدريب واحدة منها يتبيّن أنّه
قريب إلى حدٍّ لافت من التعلّم المُوجَّه (supervised learning) في الجزء الأول.

وكلّ ما عدا ذلك تملكه سلفًا — `Linear` و`Adam` و`log_softmax` و`backward()`.
بل إنّ هذا الفصل والذي يليه نقلٌ لمشروع PyTorch منفصل،
[deep-reinforcement-learning-games-from-scratch](https://github.com/amjadmajid/deep-reinforcement-learning-games-from-scratch)،
إلى BabyTorch — وهذا النقل يكاد يكون مجرّد إعادة تسمية، لأنّ BabyTorch يعكس
PyTorch. وهذا هو المغزى الضمني للجزء الثالث: إن استطعت بناء مُصنِّف، استطعت
بناء وكيل (agent).

نبقى مع **GridWorld** من الفصل 9 — كلّ خطوة تكلّف `-1`، وبلوغ الهدف يمنح
مكافأة (reward) إضافية قدرها `+10`، فلا يكون العائد (return)
`G = r₀ + γ r₁ + γ² r₂ + …` مرتفعًا إلّا لمسارٍ ناجحٍ قصير. وحلقة `reset` /
`step` نفسها تُشغِّل كلّ وكيل هنا.

## السياسة: شبكة تختار

دماغ الوكيل **سياسة**: شبكة تقرأ الحالة وتُخرج درجةً واحدة (logit) لكلّ فعل.
يحوّل softmax تلك الدرجات إلى احتمالات، ويأخذ الوكيل *عيّنةً* لفعلٍ منها —
عيّنةً لا argmax، لأنّ وكيلًا لا يجرّب جديدًا أبدًا لا يمكنه أن يكتشف أنّه
كان مخطئًا.

خطوة أخذ العيّنة والتقييم هذه هي القطعة الوحيدة التي يمنحك إيّاها PyTorch
(`torch.distributions.Categorical`) ولا يمنحكها BabyTorch. فنكتبها نحن — في
نحو اثني عشر سطرًا، كلّها من `log_softmax` والفهرسة المتقدّمة من الإنتروبيا
المتقاطعة (cross-entropy) في الفصل 3:

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>tutorials/rl/common.py</code> (التوزيع الفئوي)</summary>

```python
class Categorical:
    # ...
    def __init__(self, logits):
        self.logits = logits                     # (B, n_actions) Tensor
        self.log_probs = logits.log_softmax(axis=-1)   # (B, n_actions)

    def sample(self):
        """Return one sampled action id per row, as a NumPy int array."""
        probs = np.exp(to_numpy(self.log_probs.data))       # (B, n)
        # Inverse-CDF sampling, vectorised over the batch: draw u ~ U(0,1)
        # and take the first action whose cumulative probability exceeds it.
        u = np.random.random((probs.shape[0], 1))
        return (probs.cumsum(axis=1) > u).argmax(axis=1)

    def log_prob(self, actions):
        """Log-probabilities of ``actions`` (a 1-D int array) -> Tensor (B,)."""
        actions = xp.asarray(actions).astype(xp.int64)
        return self.log_probs[xp.arange(actions.shape[0]), actions]

    def entropy(self):
        """Entropy of each row, ``-sum(p * log p)`` -> Tensor (B,)."""
        return -(self.log_probs.exp() * self.log_probs).sum(axis=-1)
```

</details>

`log_prob` هي المهمّة: تُعيد مُوَتِّرًا لا يزال موصولًا بالمخطّط، فيستطيع
`backward()` أن يُمرِّر التدرُّجات إلى السياسة عبر الفعل الذي اختارته. تمسّك
بهذه — فهي *بعينها* تدرُّج السياسة.

## REINFORCE: اجعل الأفعال الجيّدة أكثر ترجيحًا

إليك فكرة تدرُّجات السياسة كاملةً في جملة واحدة: **العب حلقةً (episode)،
ولكلّ فعلٍ اتُّخذ، ارفع احتماله إن كان العائد جيّدًا واخفضه إن كان سيّئًا.**

![تدرُّج السياسة يحوّل حلقةً واحدة لُعِبت إلى سياسة أفضل: مسار من الحالات والأفعال والمكافآت يصبح عائدًا G-t لكلّ خطوة، وهو يرجّح الخسارة سالب log pi(a-t بمعلومية s-t) مضروبًا في الأفضلية A-t؛ ولوحة جانبية تُبيّن أنّ REINFORCE وActor-Critic وPPO لا تختلف إلّا في كيفية حساب الأفضلية](../figures/fig-policy-gradient.svg)

خسارة الفعل الواحد هي `-log π(a|s) · Aₜ`، حيث `Aₜ` هي **الأفضلية**
(advantage) — أي مدى جودة النتيجة، نسبةً إلى خطّ أساس (baseline). وتصغيرها
(بالانحدار التدرُّجي) *يرفع* لوغاريتم احتمال الأفعال ذات الأفضلية الموجبة.
إشارة الطرح تلك هي الحيلة كلّها: تحوّل «تعظيم المكافأة» إلى خسارة يستطيع
`backward()` أن يقضمها.

يستخدم REINFORCE أبسط خطّ أساس ممكن — متوسّط العائد على دُفعة من الحلقات —
فلا يسأل `Aₜ` إلّا: «هل تفوّق هذا الفعل على فعلٍ اعتيادي؟». والتحديث في أربعة
أسطر:

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>tutorials/rl/reinforce.py</code> (خطوة تدرُّج السياسة)</summary>

```python
        states = np.concatenate(b_states)
        actions = np.concatenate(b_actions)
        advantages = C.to_tensor(C.normalize(np.concatenate(b_returns)))

        # --- one policy-gradient step --------------------------------------
        dist = C.Categorical(policy(C.to_tensor(states)))
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * advantages).mean() - args.ent_coef * dist.entropy().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

</details>

حدّ `entropy` رشوة لطيفة تُبقي الاستكشاف قائمًا: يكافئ السياسة على بقائها
غير متيقّنة قليلًا، حتى لا تنهار على فعلٍ واحد قبل أن تتعلّم حقًّا. شغِّله،
فيرتقي الوكيل من التخبّط (عائد ≈ −100) إلى حلّ المتاهة في كلّ مرّة (عائد ≈
+2) في أقلّ من مئة تحديث.

## Actor-Critic: تعلَّم خطّ الأساس

خطّ أساس REINFORCE رقم واحد للدُّفعة كلّها. لكنّ حالةً قرب الهدف أفضل *حقًّا*
من أخرى في ركن بعيد، وخطّ أساس مسطّح لا يفرّق بينهما — فتبقى الأفضلية
مشوَّشة، ويكون التعلّم أبطأ ممّا ينبغي.

العلاج شبكة ثانية، **الناقد** (critic) `V(s)`، تُدرَّب لتتنبّأ بالعائد من
كلّ حالة. فتصير الأفضلية `return − V(s)`: «كم كان هذا أفضل من المتوقَّع؟»،
مُقيَّمةً لكلّ حالة على حدة. شبكتان، وخسارتان، وجولة (rollout) واحدة مشتركة:

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>tutorials/rl/actor_critic.py</code> (الأفضلية، ثمّ تحديثان)</summary>

```python
        values = critic(states)                              # V(s), with grad
        # Advantage is a *fixed weight* for the actor, so compute it in NumPy
        # (this is the "detach the critic" step): how much each return beat
        # the critic's expectation, then normalised for a steady gradient.
        advantages = C.to_tensor(C.normalize(returns - to_numpy(values.data)))

        # --- actor update: policy gradient, weighted by advantage ----------
        dist = C.Categorical(actor(states))
        actor_loss = (-(dist.log_prob(actions) * advantages).mean()
                      - ent_coef * dist.entropy().mean())
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        # --- critic update: regress V(s) toward the actual returns ---------
        critic_loss = ((C.to_tensor(returns) - values) ** 2).mean()
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()
```

</details>

يُدرَّب الناقد بانحدارٍ بسيط (regression) — متوسّط مربّع الخطأ (MSE) مقابل
العوائد التي يحاول التنبّؤ بها، وهي عينها الخسارة من الفصل 3. أمّا الفاعل
(actor) فيستخدم حُكم الناقد خطَّ أساسٍ أذكى. تدرُّج السياسة نفسه؛ لكن بضوضاء
أقلّ.

## PPO: ما يستخدمه الجميع فعلًا

Actor-Critic يعمل لكنّه مُبذِّر وهشّ: كلّ دُفعة من الخبرة المكتسبة بعناء
تُشغِّل خطوة تدرُّج واحدة ثمّ تُرمى، ولا شيء يمنع تلك الخطوة من أن تكون كبيرة
إلى حدٍّ يدمّر السياسة.

يُعالج **تحسين السياسة القريب** (Proximal Policy Optimization) كليهما، وهو
العمود الفقري لمعظم التعلّم المُعزَّز الحديث — بما فيه RLHF الذي يوائم نماذج
اللغة («المكافأة» هناك نموذجُ تفضيلٍ بشري، لكنّ الخوارزمية هي هذه). حيلته هي
**الهدف المقصوص** (clipped objective). تتبّع النسبة بين السياسة الجديدة وتلك
التي جمعت البيانات، `r = π_new / π_old`، واقصصها إلى `[1−ε, 1+ε]` حتى لا
يستطيع تحديث واحد أن يزيح السياسة بعيدًا أكثر من اللازم:

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>tutorials/rl/ppo.py</code> (الهدف البديل المقصوص)</summary>

```python
            dist = C.Categorical(actor(s))
            new_logp = dist.log_prob(actions[idx])
            # ratio = pi_new / pi_old, computed in log-space for stability.
            ratio = (new_logp - C.to_tensor(old_logps[idx])).exp()
            adv = C.to_tensor(advantages[idx])

            # The clipped surrogate: take the *pessimistic* of the raw and
            # clipped objectives, so an update that would move the policy
            # outside [1-eps, 1+eps] earns no extra reward.
            unclipped = ratio * adv
            clipped = C.clip_tensor(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -C.minimum(unclipped, clipped).mean()
```

</details>

حاجز الأمان ذاك هو ما يتيح لـ PPO أن **يُعيد استخدام كلّ دُفعة لعدّة حِقَب**
(epochs) من تحديثات الدُّفعات الصغيرة بدل واحدة — وهو أكفأ بكثير في استهلاك
العيّنات. (`clip_tensor` و`minimum` تعملان عنصرًا بعنصر، وقابلتان للاشتقاق،
ومبنيّتان من `relu` — إذ إنّ `Tensor.max` في BabyTorch لا يختزل إلّا على
امتداد محور، فيصنع [`common.py`](../../tutorials/rl/common.py) النسختين
العاملتين عنصرًا بعنصر بالطريقة نفسها التي بنى بها الفصل 2 كلّ شيء آخر.) وعلى
GridWorld هو الفائز بلا منازع: يُحلّ في نحو 15 تحديثًا، بينما يحتاج REINFORCE
إلى نحو 70.

## شغِّله

```bash
cd tutorials/rl
python reinforce.py        # policy gradient
python actor_critic.py     # + a learned baseline
python ppo.py              # clipped, sample-efficient
python reinforce.py --render_every 20     # watch it think, in ASCII
```

يطبع كلٌّ منها تقدّمه، ويحفظ منحنى تعلُّم، ويتقاسم درسًا واحدًا: «الوكيل»
شبكة تُدرَّب بخسارة غير اعتيادية بعض الشيء. يحوي
[README الدرس](../../tutorials/rl/README.md) الجدول الكامل؛ ويسلك الفصل
التالي الطريق *الآخر* — تعلُّم القيم بدل السياسة — ويستخدمه للعب Snake.

## تمارين

**اختبر فهمك** (الإجابات تُفتَح):

**س1.** خسارة REINFORCE هي `-(log_prob * advantage)`. إن قاد فعلٌ إلى عائدٍ
*دون المتوسّط* (أفضلية سالبة)، فبأيّ اتّجاه يحرّك التحديث احتمال ذلك الفعل،
ولماذا؟

<details><summary>الإجابة</summary>

إلى الأسفل. مع أفضلية سالبة، *يُصغَّر* `-(log_prob · advantage)` بجعل
`log_prob` **أصغر** (أقلّ سلبيّةً في الإجمال)، فيخفض الانحدار التدرُّجي
احتمال ذلك الفعل. فالأفعال ذات النتيجة الجيّدة (أفضلية موجبة) تُدفَع إلى
الأعلى، والسيّئة إلى الأسفل — وهو المقصود بالضبط.

</details>

**س2.** لماذا نأخذ عيّنةً من السياسة أثناء التدريب بدل اتّخاذ الفعل الأرجح
دائمًا (argmax)؟

<details><summary>الإجابة</summary>

الاستكشاف. الوكيل الذي يتّخذ دائمًا فعله الأفضل حاليًّا لا يجرّب البدائل
أبدًا، فلا يمكنه أن يكتشف أنّ فعلًا آخر كان أفضل فعليًّا — فيعلق. أخذ
العيّنات (إضافةً إلى علاوة الإنتروبيا) يُبقيه يجرّب الأشياء. أمّا عند
*التقييم*، بعد اكتمال التعلّم، فتنتقل إلى argmax الجَشِع.

</details>

**س3.** يحسب كلٌّ من Actor-Critic وPPO أفضليّةً. ما الذي يتعطّل إن أدخلت
تدرُّج *الناقد* في خسارة الفاعل (أي نسيت أن تعامل الأفضلية بوصفها ثابتًا)؟

<details><summary>الإجابة</summary>

يُقصَد بالأفضلية أن تكون *وزنًا* ثابتًا على تدرُّج السياسة، لا شيئًا يُحسَّن.
فإن تدفّقت التدرُّجات عبرها إلى الناقد، بدأ تحديثُ الفاعل يدفع الناقدَ إلى
جعل الأفضليّات تبدو جيّدة بدل التنبّؤ بالعوائد — فيكفّ خطّ الأساس عن أن يكون
خطّ أساسٍ أمين، ويصبح التدريب غير مستقرّ. ولهذا تحسب الشيفرة الأفضلية في
NumPy (أو كانت ستستدعي `.detach()` عليها): إيقافٌ متعمَّد للتدرُّج.

</details>

**طبِّقه بنفسك** — يحتاج الوكيل القائم على القيمة في الفصل التالي إلى خسارة
Huber ومخزن إعادة (replay buffer)؛ جرّب تخطيط `discounted_returns`
و`compute_gae` انطلاقًا من سلاسل توثيقهما في
[`tutorials/rl/common.py`](../../tutorials/rl/common.py) قبل أن تقرأهما.

---

**الشيفرة:**
[`tutorials/rl/common.py`](../../tutorials/rl/common.py) ·
[`tutorials/rl/gridworld.py`](../../tutorials/rl/gridworld.py) ·
[`tutorials/rl/reinforce.py`](../../tutorials/rl/reinforce.py) ·
[`tutorials/rl/actor_critic.py`](../../tutorials/rl/actor_critic.py) ·
[`tutorials/rl/ppo.py`](../../tutorials/rl/ppo.py)

[→ الفصل 9: التعلّم المُعزَّز الجدولي](09-tabular-methods.md) | [المحتويات](README.md) | [الفصل 11: تعلُّم Q العميق ←](11-deep-q-learning.md)
