# الفصل 8 — تدريب نموذج GPT

*الجزء الثاني، الفصل 4 من 4. النموذج مبنيّ. والآن نُمرِّره عبر دورة الحياة
الكاملة لنموذجٍ لغويّ — نُدرِّبه تدريبًا مُسبقًا على مُدوَّنة، ونُولِّد نصًّا،
ونضبطه بدقّة على صوتٍ جديد — مستعينين بالسكربتات القابلة للتشغيل في
[`tutorials/llm/`](../../tutorials/llm/).*

## التدريب المُسبق: الحلقة تلتقي جبلًا من النصّ

تُقرأ [`train.py`](../../tutorials/llm/train.py) من أعلاها إلى أسفلها كخلاصةٍ
للكتاب بأكمله:

```python
text = load_tiny_shakespeare()            # ~1.1 MB of raw text   (ch. 5)
tokenizer = CharTokenizer(text)           # vocab ≈ 65 characters (ch. 5)
data = encode_corpus(text, tokenizer)     # one long id array     (ch. 5)
train_data, val_data = data[:n], data[n:] # 90/10 honesty split   (ch. 4)

model = GPT(vocab_size, block_size=128,   # the Transformer       (ch. 6-7)
            n_embd=192, n_head=6, n_layer=6, dropout=0.1)
optimizer = AdamW(model.parameters(), learning_rate=3e-3,
                  weight_decay=0.1)       # the GPT optimizer     (ch. 4)
scheduler = CosineWarmupLR(optimizer, warmup_steps, total_steps)

for step in range(steps):
    scheduler.step(step)
    x, y = get_batch(train_data, block_size, batch_size)  # (B,T) windows
    loss = model.loss(x, y)               # forward + cross-entropy
    optimizer.zero_grad()
    loss.backward()                       # ch. 2 does the rest
    optimizer.step()
```

لا شيء هنا جديد — وتلك هي الفكرة. "التدريب المُسبق" ليس إجراءً خاصًّا؛ إنّه حلقة
الخطوات الأربع من الفصل 4، تُغذَّى بدُفعاتٍ ذاتيّة الإشراف من الفصل 5، وتُحدِّث
نموذج الفصل 7. الجديد الوحيد هو *حجم الطموح*: استيعاب إحصاءات لغةٍ ما داخل
الأوزان، دون أيّ تسميات سوى النصّ نفسه.

## قراءة منحنى الخسارة

الإنتروبيا المتقاطعة (cross-entropy) تمنح التدريب مقياسًا ذا معنى. مع 65 محرفًا
متساوية الاحتمال، تكون خسارة نموذجٍ بلا أدنى فكرة `−ln(1/65) ≈ 4.17` — وهذا
بالضبط حيث يبدأ المنحنى. وكلّما هبطت، تسلّقت العيّنات سُلَّمًا من الكفاءة:

```
loss ~4.2   random keysmash        "xQj;wRk?vB"
loss ~3.0   letter frequencies     "e soaet htn re"
loss ~2.5   word-shaped strings    "ther sonot hind"
loss ~2.0   words, some grammar    "the king hath sent"
loss ~1.5   dialogue that scans    "ROMEO: What say you?"
```

اطبع خسارة التدريب وخسارة التحقّق معًا (السكربت يفعل ذلك كلّ 100 خطوة): ما دامتا
تهبطان معًا فالنموذج يتعلّم شكسبير؛ وحين تستمرّ خسارة التدريب في الهبوط بينما
ترتفع خسارة التحقّق، يكون قد بدأ يحفظ المُدوَّنة عن ظهر قلب — إنّه فرط التخصيص من
الفصل 4، حيًّا أمامك.

## التوليد: الحلقة التي تكتب

النموذج المُدرَّب يربط السياق باحتمالات الرمز التالي. والكتابة ليست إلّا تطبيقه
مرارًا وتكرارًا — `GPT.generate` في [`model.py`](../../tutorials/llm/model.py):

![التوليد مُصنِّف داخل حلقة: يُمرَّر السياق (المقصوص إلى block_size) عبر نموذج GPT المُجمَّد ليُنتج قيم logits، فتُقسَم على درجة الحرارة، وتُصفّى إلى top-k، وتُحوَّل إلى احتمالات بواسطة softmax، ثمّ تُؤخذ منها عيّنة؛ ويُضاف الرمز الجديد وتدور الحلقة من جديد — رمزٌ واحد في كلّ لفّة](../figures/fig-generation.svg)

ثلاث تفاصيل عمليّة، جميعها ظاهرة في الشيفرة:

* **قصّ السياق.** لا يستطيع النموذج أن يلتفت إلّا إلى `block_size` رمزًا، فيُغذّي
  التوليد `idx[:, -block_size:]` — نافذة مُنزلِقة. (تضمينات المواضع موجودة حتى
  `block_size` فقط؛ وما وراء ذلك يسقط النصّ الأقدم ببساطة خارج مجال الرؤية.)
* **`no_grad` + `eval`.** لن يتبع ذلك أيّ تمرير خلفيّ، فتسجيل المخطّط (الفصل 2)
  سيُهدر الذاكرة؛ ويجب إيقاف dropout.
* **أخذ العيّنات، لا argmax.** اختيار المحرف الأرجح وحده في كلّ مرّة يحبس النموذج
  في حلقاتٍ مُكرَّرة. نحن *نأخذ عيّنة* من التوزيع — ونتحكّم في العشوائية.

**درجة الحرارة** تُعيد تحجيم قيم logits قبل softmax
(`logits / temperature`):

| درجة الحرارة | الأثر |
|-------------|--------|
| → 0 | argmax: أأمن رمزٍ في كلّ مرّة؛ وسرعان ما يقع في تكرارٍ رتيب |
| 0.7–0.9 | أحدّ من المُتعلَّم: متماسك، ومحافظ قليلًا |
| 1.0 | توزيع النموذج المُتعلَّم الصادق |
| > 1.2 | مُسطَّح: إبداعيّ ينزلق نحو خبطٍ عشوائيّ للوحة المفاتيح |

**top-k** هو الحارس المُكمِّل: أبقِ على الرموز الأرجح البالغ عددها `k` فقط،
وصفِّر البقيّة، وأعِد التطبيع. إنّه يقطع الذيل الطويل من الرموز الرديئة غير
المُرجَّحة فُرادى، والتي يكفي احتمالها *المُجتمِع* لأن يُخرج الجملة عن مسارها.

**ذاكرة KV المؤقّتة** تزيل الهدر الخفيّ في التوليد. تحتاج كلّ لفّة من الحلقة إلى
صفٍّ واحد من logits — الأحدث — لكنّ التمرير الأماميّ الساذج يُعيد حساب الانتباه
على السياق *كامله* لإنتاجه: تُعاد مفاتيح كلّ موضعٍ قديم وقيمه من الصفر، لفّةً بعد
لفّة، لتخرج متطابقة في كلّ مرّة. لذا تحتفظ بها `generate`. يُمرَّر المُوجِّه
(prompt) مرّةً واحدة وتُخزِّن كلّ كتلة مصفوفتَي `k` و`v` الخاصّتين بها (**التعبئة
المُسبقة**)؛ وبعد ذلك تُمرّر كلّ لفّة **الرمز الذي أُخذت عيّنته في اللفّة السابقة
فقط**، وتحسب قيم q وk وv الجديدة الوحيدة الخاصّة به، وتلتفت إلى الماضي المُخزَّن.
نفس logits، وجزءٌ يسير من العمل — كلّ نموذج لغويّ كبير في الإنتاج يخدمك عبر ذاكرة
مؤقّتة كهذه.

تستحقّ تفصيلتان من الذاكرة المؤقّتة القراءة في الشيفرة. الاستعلام الجديد الوحيد
لا يحتاج إلى أيّ إخفاء: كلّ موضعٍ مُخزَّن *هو* ماضيه (صفّ القناع الذي يقتطعه هو
الأخير — أصفارٌ كلّه). وحين يتجاوز السياق `block_size`، تنزلق النافذة؛ وتضميناتنا
للمواضع مُطلَقة، فيصبح كلّ مفتاحٍ مُخزَّن فجأةً تابعًا لمُعرّف موضعٍ انزاح، فتُعاد
بناء الذاكرة المؤقّتة البائتة. تُوقِّت `generate.py` نفسها زمنيًّا — شغّلها بـ
`--no_cache` لتشعر بما تشتريه الذاكرة المؤقّتة.

<details>
<summary><b>كيف يُنفَّذ ذلك</b> — <code>tutorials/llm/model.py</code> (حلقة الكتابة، دون اختصار)</summary>

```python
        idx = xp.asarray(idx).astype(xp.int64)
        if idx.ndim == 1:
            idx = idx[None, :]

        self.eval()
        caches = self.empty_kv_caches() if use_cache else None
        with babytorch.no_grad():
            for _ in range(max_new_tokens):
                # Never use more than block_size tokens of context.
                idx_cond = idx[:, -self.block_size:]

                if caches is None:
                    # No cache: forward the whole context, every lap.
                    logits = self.forward(idx_cond).data
                elif caches[0]["k"] is None:
                    # First lap ("prefill"): run the whole prompt once and
                    # let every block store its keys and values.
                    logits = self.forward(idx_cond, caches).data
                elif caches[0]["k"].shape[2] < self.block_size:
                    # Steady state: the past is cached -- forward only the
                    # token sampled on the previous lap.
                    logits = self.forward(idx[:, -1:], caches).data
                else:
                    # The context window is full and now slides one step
                    # per token.  Our position embeddings are absolute, so
                    # every cached key belongs to a position id that just
                    # shifted -- the cache is stale; rebuild it.
                    caches = self.empty_kv_caches()
                    logits = self.forward(idx_cond, caches).data

                logits = logits[:, -1, :] / temperature   # last position: (B, vocab)

                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    # Zero out everything below the k-th largest logit.
                    kth = xp.sort(logits, axis=-1)[:, -k][:, None]
                    logits = xp.where(logits < kth, -xp.inf, logits)

                # Softmax -> probabilities, then sample one token per row.
                probs = _softmax_np(logits)
                next_id = _sample(probs)                  # (B, 1)
                idx = xp.concatenate([idx, next_id], axis=1)
        return idx
```

</details>

## الضبط الدقيق: نفس الحلقة، صوت جديد

[`finetune.py`](../../tutorials/llm/finetune.py) هي `train.py` مع ثلاثة أسطرٍ
مُغيَّرة: **حمِّل نقطة الحفظ المُدرَّبة مُسبقًا بدل الأوزان العشوائية، واستخدم
معدّل تعلّمٍ أصغر بكثير، وغذِّ مُدوَّنةً مختلفة** (أناشيد الأطفال). وبعد بضع مئات
من الخطوات يكتب النموذج أناشيد — مستخدمًا ما تعلّمه من شكسبير من هجاءٍ وإيقاعٍ
وبنية حوار.

ذلك النقل الرخيص هو أعمق حقيقة في النصف الثاني من هذا الكتاب. الأوزان المُدرَّبة
مُسبقًا ليست جدول بحثٍ لشكسبير؛ إنّها آليّة قابلة لإعادة الاستخدام لأجل *نصٍّ على
هيئة الإنجليزية*، ودفعةٌ صغيرة تُعيد توظيفها. وسِّع الوصفة نفسها، وهكذا يُصنَع كلّ
نموذجٍ لغويّ كبير منشور: درِّبه تدريبًا مُسبقًا مرّةً واحدة على الإنترنت، ثمّ
كيّفه بثمنٍ زهيد — للمحادثة، وللبرمجة، ولمستنداتك. (لكن اضبط بدقّة مدّةً أطول من
اللازم على المُدوَّنة الصغيرة، فتنجرف المهارات القديمة بعيدًا — *النسيان الكارثيّ*:
راقبه يحدث بأن تُولِّد باستخدام مُوجِّهٍ شكسبيريّ بعد خطوات ضبطٍ كثيرة.)

نقاط الحفظ تجعل سير العمل هذا مُمكنًا — تكتب `save_checkpoint`
([`common.py`](../../tutorials/llm/common.py)) ثلاثة ملفّات: الأوزان (`Module.save`
من الفصل 3، آمنة على الـ GPU)، ومُفردات المُجزِّئ، وإعدادات النموذج، حتى تستطيع
`load_checkpoint` أن تُعيد بناء نموذجٍ مطابق في أيّ مكان.

## شغّله

```bash
cd tutorials/llm
python train.py --steps 3000                    # pretrain (GPU: minutes)
python generate.py --prompt "ROMEO:" --tokens 400 --temperature 0.8
python attention_viz.py                         # draw what the heads learned
python finetune.py                              # adapt to nursery rhymes
python generate.py --checkpoint checkpoints/babygpt_finetuned --prompt "Twinkle"
```

على المعالج، صغِّر النموذج — سيظلّ يتسلّق السُّلَّم نفسه، لكن إلى درجةٍ أدنى:

```bash
python train.py --steps 1500 --block_size 64 --n_embd 96 --n_head 4 --n_layer 4
```

ثمّ جرِّب. من أفضل التجارب الأولى: استبدل `CharTokenizer` بـ
[`BPETokenizer`](../../babytorch/text/tokenizers.py) وقارن الخسارة *لكلّ محرف*
(وحداتٌ مُنصفة — إذ يتنبّأ BPE بنصٍّ أكثر لكلّ رمز)؛ درِّب على مُدوَّنتك أنت
`--corpus file.txt`؛ ارسم أوزان الانتباه `(T, T)` لرؤوسك المُدرَّبة بـ
[`attention_viz.py`](../../tutorials/llm/attention_viz.py) وابحث عن بنية — قُطر
الرمز السابق، وعمودٌ ساطع عند سطرٍ جديد (ثمّ اقرأ السكربت: إنّه مُجرّد `att.data`
من الفصل 6، مُحتفَظ به براية)؛ وادفع `temperature` إلى أقصى الحدود وراقب السُّلَّم
بالمقلوب.

## حيث ينتهي BabyGPT

بين هذا النموذج ذي الـ 2.7 مليون مُعامِل ونموذجٍ لغويّ كبير من الطليعة يقع،
بصراحة: نحو خمس رُتبٍ من المقدار في الحجم (المُعامِلات، والبيانات، والحوسبة)؛
وهندسةٌ لأجل ذلك الحجم (نوى GPU مُندمجة، ودقّة مختلطة، وخدمة مُدفَّعة، وتدريب
مُجزَّأ عبر آلاف الأجهزة)؛ والتدريب اللاحق — الضبط بالتعليمات والتعلّم المُعزَّز من
التغذية الراجعة البشرية، وهو ضبطٌ دقيق (من النوع الذي فعلته للتوّ) موجَّه نحو "كن
مفيدًا" بدل "اُنظم قافية".

أمّا ما **لا** يتغيّر فهو كلّ ما غطّاه هذا الكتاب: المُوَتِّرات، والاشتقاق
التلقائي، والإنتروبيا المتقاطعة، وAdamW، والانتباه، والكتل المتبقّية، وأخذ عيّنة
الرمز التالي. افتح أيّ تنفيذٍ جادّ — لِنقُل شيفرة PyTorch لمُحوِّلٍ في الإنتاج —
وستتعرّف على كلّ جزء، لأنّك قد قرأت الآن واحدًا كاملًا.

احذف كلمة "baby" وواصِل المسير.

## تمارين

**اختبر فهمك** (الإجابات تُفتَح بالنقر):

**س1.** نموذجٌ على مستوى المحرف بمُفردات من 65 رمزًا يبدأ التدريب عند خسارة ≈
4.17. من أين يأتي ذلك الرقم؟

<details><summary>الإجابة</summary>

`−ln(1/65)`. الأوزان غير المُدرَّبة توزّع الاحتمال بانتظامٍ تقريبًا، فيحظى المحرف
التالي الحقيقيّ بـ `p ≈ 1/65`، وتُحمّله الإنتروبيا المتقاطعة لوغاريتمه السالب.
وأيّ خسارةٍ بدائيّة بعيدة عن `ln(vocab_size)` علامةٌ على وجود خلل.

</details>

**س2.** إحدى العيّنات تقول `"the the the the the"`، وأخرى `"xq;Rd wke,pf"`. أيّهما
جاءت من درجة حرارة 0.1 وأيّهما من 1.8؟

<details><summary>الإجابة</summary>

التكرار هو T = 0.1: زيادة الحدّة تدفع أخذ العيّنات نحو argmax، والفكّ شبه الجَشِع
يقع في حلقات. والخبط العشوائيّ هو T = 1.8: التسطيح يمنح الرموز الرديئة احتمالًا
حقيقيًّا. والنصّ القابل للاستخدام يعيش بينهما — ولهذا 0.8 هي القيمة الافتراضية.

</details>

**س3.** لماذا يستخدم الضبط الدقيق معدّل تعلّمٍ أصغر بنحو 10× من التدريب المُسبق؟

<details><summary>الإجابة</summary>

الأوزان تُرمِّز شيئًا ثمينًا بالفعل؛ والضبط الدقيق ينبغي أن *يدفعها* نحو المُدوَّنة
الجديدة، لا أن يكتب فوقها. والمعدّل المُفرط في الكِبَر يمحو المعرفة المُدرَّبة
مُسبقًا — نسيانٌ كارثيّ، وبسرعة.

</details>

**س4.** تحمل ذاكرة KV المؤقّتة 10 مواضع، وتُمرّر `generate` الرمز الواحد الذي
أخذت عيّنته للتوّ. داخل كلّ كتلة، ما شكل `q` وشكل جدول الانتباه `att` — ولماذا
تبقى السببيّة قائمة دون إخفاء أيّ شيء؟

<details><summary>الإجابة</summary>

`q` هو `(B, n_head, 1, head_size)` — استعلامٌ واحد — و`att` هو
`(B, n_head, 1, 11)`: صفٌّ واحد من الأوزان على المواضع العشرة المُخزَّنة زائد
الرمز نفسه. لا شيء يحتاج إلى إخفاء لأنّ كلّ ما في الذاكرة المؤقّتة هو ماضي هذا
الموضع؛ والصفّ الذي تقتطعه الشيفرة من القناع هو الأخير، وهو أصفارٌ كلّه.

</details>

**طبِّقه بنفسك** — نفِّذ `top_p_filter` (أخذ العيّنات بالنواة (nucleus): أبقِ على
مقدارٍ ثابت من *الاحتمال* بدل *العدد* الثابت في top-k) و★ `generate_greedy` (ثمّ
راقب النصّ الجَشِع يدور في حلقة — سبب لجوئنا إلى أخذ العيّنات)، في
[`exercises/ch08_generation.py`](../exercises/ch08_generation.py)؛ ثمّ شغّل
`pytest book/exercises/test_ch08_generation.py -v`.
([كيف تعمل التمارين](../exercises/README.md).)

---

**ملفّات المصدر لهذا الفصل:**
[`tutorials/llm/train.py`](../../tutorials/llm/train.py) ·
[`tutorials/llm/generate.py`](../../tutorials/llm/generate.py) ·
[`tutorials/llm/finetune.py`](../../tutorials/llm/finetune.py) ·
[`tutorials/llm/attention_viz.py`](../../tutorials/llm/attention_viz.py) ·
[`tutorials/llm/common.py`](../../tutorials/llm/common.py)

[→ الفصل 7: المُحوِّل](07-transformer.md) | [المحتويات](README.md) | [الفصل 9: التعلّم المُعزَّز الجدوليّ ←](09-tabular-methods.md)
