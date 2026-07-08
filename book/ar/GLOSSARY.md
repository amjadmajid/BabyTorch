# مسرد المصطلحات ودليل الترجمة — BabyTorch

هذا المسرد يوحّد ترجمة المصطلحات التقنية عبر الفصول والأشكال، حتى تبقى النسخة
العربية متّسقة. القاعدة العامة: نترجم النثر، ونُبقي الشيفرة كما هي.

## قواعد الترجمة (مهمّة)

1. **الشيفرة لا تُترجَم.** كلّ ما هو داخل كتل ``` ``` ``` ``` ``` أو داخل
   `` `code` `` أو مسارات الملفّات `<code>path</code>` يبقى **حرفيًّا** كما في
   المصدر الإنجليزي — لأنّ اختبار `tests/test_book_snippets.py` يتحقّق من تطابق
   الشيفرة حرفًا بحرف. تُترجَم فقط: النثر، والعناوين، ونصّ الجداول، ونصّ وصف
   الأشكال (alt)، والعبارة الوصفيّة في `<summary>`.
2. **الرموز الرياضية** (ε، βₜ، √، →، ×، ≈) تبقى كما هي.
3. **إعادة كتابة الروابط** (لأنّ الملفّات العربية أعمق بمستوى واحد):
   - روابط ملفّات المستودع `../babytorch/…` و`../tutorials/…` و`../tests/…`
     و`../README.md` ← تصبح `../../…`.
   - التمارين المُقيَّمة `exercises/chNN_*.py` ← تصبح `../exercises/chNN_*.py`
     (نُعيد استخدام شيفرة `book/exercises/` المشتركة، لا نكرّرها).
   - روابط الأشكال ← تصبح `../figures/fig-*.svg`: النسخة العربية تعيد استخدام
     أشكال النسخة الإنجليزية نفسها (بتسمياتها الإنجليزية)، فلا تُترجَم الأشكال.
   - روابط الفصول و`README.md` تبقى كما هي، مع **قلب اتّجاه الأسهم** في الترويسة
     السفلية: "التالي" يشير إلى اليمين والسابق إلى اليسار في السياق العربي.
4. عند أوّل ذكر لمصطلح مفتاحي، نضع المقابل الإنجليزي بين قوسين.

## المصطلحات

| الإنجليزية | العربية |
|---|---|
| deep learning | التعلّم العميق |
| machine learning | تعلّم الآلة |
| neural network | الشبكة العصبية |
| framework | إطار العمل |
| tensor | المُوَتِّر (tensor) |
| array | المصفوفة (array) |
| shape | الشكل (shape) |
| dimension / axis | البُعد / المحور |
| broadcasting | البثّ (broadcasting) |
| matrix multiplication | ضرب المصفوفات |
| batch | الدُّفعة (batch) |
| gradient | التدرُّج (gradient) |
| gradient descent | الانحدار التدرُّجي |
| backpropagation | الانتشار العكسي (backpropagation) |
| forward / backward pass | التمرير الأمامي / الخلفي |
| computation graph | مخطّط الحساب |
| chain rule | قاعدة السلسلة |
| autograd | الاشتقاق التلقائي (autograd) |
| loss / loss function | الخسارة / دالّة الخسارة |
| optimizer | المُحسِّن (optimizer) |
| learning rate | معدّل التعلّم |
| training | التدريب |
| model | النموذج |
| weights / parameters | الأوزان / المُعامِلات |
| layer | الطبقة |
| activation function | دالّة التنشيط |
| embedding | التضمين (embedding) |
| tokenization / token | التجزئة إلى رموز (tokenization) / الرمز |
| vocabulary | المُفردات |
| attention / self-attention | الانتباه (attention) / الانتباه الذاتي |
| transformer | المُحوِّل (Transformer) |
| head (attention) | الرأس |
| residual connection | الوصلة المتبقّية (residual) |
| normalization / LayerNorm | التسوية / تسوية الطبقة (LayerNorm) |
| convolution | الالتفاف (convolution) |
| pooling | التجميع (pooling) |
| upsampling | رفع الدقّة (upsampling) |
| softmax | softmax |
| probability / distribution | الاحتمال / التوزيع |
| inference | الاستدلال |
| generation / sampling | التوليد / أخذ العيّنات |
| reinforcement learning | التعلّم المُعزَّز (reinforcement learning) |
| agent / environment | الوكيل / البيئة |
| reward / return | المكافأة / العائد |
| policy | السياسة (policy) |
| value function | دالّة القيمة |
| state / action | الحالة / الفعل |
| diffusion (model) | الانتشار (diffusion) |
| noise / denoise | الضوضاء / إزالة الضوضاء |
| noise schedule | جدول الضوضاء |
| U-Net | شبكة U-Net |
| CPU / GPU | المعالج / وحدة معالجة الرسوميّات (GPU) |
