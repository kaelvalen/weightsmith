# Görsel ML Mimari Editörü — Proje Planı

Oct 18, 2026 · @Arda

## Konumlandırma ve kapsam

Proje, mimari deneyi yapan araştırmacı için görsel model programlama ortamı. Canvas PyTorch'un üstünde bir görünüm; kaynak gerçeği graph IR. Genel kullanıcı ürünü değil.

- **Hedef kullanıcı:** özel mimari deneyen araştırmacı/mühendis. İlk kullanıcı sensin.
- **Vaat:** graph kur, doğrula, okunabilir PyTorch üret, local GPU'da eğit, run'ları karşılaştır.
- **Ayrışma noktaları:** custom modül node'u (Python escape hatch + shape kontratı), LoRA/adapter'ın modifier olarak modellenmesi ve kısmi freeze, aynı graph'ın varyantlarını diff'le karşılaştırma, pretrained modeli graph'a açma.
- **Kapsam dışı:** PyTorch dışı backend, "Run deyince model çıkar" genel kullanıcı akışı, AutoML ve hiperparametre arama, çok node'lu dağıtık eğitim (v1), marketplace.

"Genel kullanıcı" kapsamı bilinçli olarak dışarıda: hiperparametre yargısı gerektiren bir işi tek tuşa indirmek vaadi tutmaz ve araştırmacıya hizmet eden tasarım kararlarını (açıklık, export edilebilirlik) bozar.

## Mimari genel bakış

Üç katman var, tek sözleşme: Graph IR. UI IR'ı yazar, backend IR'ı okuyup kod üretir, runtime o kodu job spec ile çalıştırır. UI PyTorch'a doğrudan dokunmaz.

&#91;embedded content: mimari · 3 katman, tek sözleşme\]

Canvas, Python node ve importer IR'a yazar. Backend ve runtime yalnızca IR'ı okur. Runtime'ın metrik akışı canvas'a geri döner, her run deney deposuna IR hash'iyle kaydedilir.

## Graph IR

IR, framework bağımsız ve versiyonlu bir JSON şeması. Canvas onu çizer, backend onu derler, runtime onun hash'ini run kimliği olarak kullanır. Shape inference IR üstünde çalışır, UI'da değil.

- **Node:** `id`, `type`, `params`, tipli portlar, `adapt[]` (modifier listesi), `trainable` bayrağı.
- **Port:** ad, dtype, sembolik shape (`B`, `T`, `D` gibi boyut değişkenleri). Bağlantı anında shape ve dtype uyumu denetlenir, hata ilgili porta işlenir.
- **Edge:** `node.port → node.port`. Dataflow dışında hiçbir şey edge olmaz.
- **Subgraph:** tekrar eden bloklar (ör. N× decoder block) parametreli alt graph olarak tanımlanır, `repeat` sayısı ve opsiyonel weight tying ile.
- **Layout:** x/y koordinatları ayrı `layout` alanında tutulur ve hash'e girmez. Node'u sürüklemek yeni run üretmemeli.
- **Canonical form:** node'lar ve edge'ler sıralanmış, varsayılanlar açılmış hâlde serileştirilir. İçerik hash'i buradan çıkar.
- **Versiyonlama:** `schema` alanı + sürümler arası migration fonksiyonları. Eski proje dosyaları açılabilir kalır.

Örnek: `Input → Linear → GELU → Linear → CrossEntropy`, ilk Linear'a LoRA bağlı.

```json
{
  "schema": 1,
  "nodes": [
    {"id": "in",   "type": "Input", "params": {"shape": ["B", 768], "dtype": "f32"}},
    {"id": "fc1",  "type": "Linear", "params": {"in": 768, "out": 1024},
     "adapt": [{"kind": "lora", "target": "weight", "r": 16, "alpha": 32}]},
    {"id": "act",  "type": "GELU"},
    {"id": "fc2",  "type": "Linear", "params": {"in": 1024, "out": 10}},
    {"id": "loss", "type": "CrossEntropy"}
  ],
  "edges": [["in.out", "fc1.in"], ["fc1.out", "act.in"],
            ["act.out", "fc2.in"], ["fc2.out", "loss.logits"]]
}
```

## Adaptation modeli

LoRA dataflow'a adım eklemez, bir modülün forward'ını sarar (`Wx + BAx`). Bu yüzden node değil, node'a bağlı modifier. Canvas'ta node üzerinde rozet, edge yok.

| Alan | Anlamı |
| --- | --- |
| `kind` | `lora`, `adapter`, `custom` (kullanıcı modülü) |
| `target` | Hedef parametre veya alt modül (`weight`, `q_proj`, `v_proj`) |
| `r`, `alpha`, `dropout` | LoRA hiperparametreleri |
| `freeze` | Taban parametreler dondurulur mu, hangi maske |
| `init` | A/B başlangıcı (B=0 varsayılan) |

- **Toplu seçim:** "tüm attention q/v'ye LoRA" tek hamlede. Seçici bir sorgu (tip + isim deseni) node listesine uygulanır, sonuç tek tek annotation olarak yazılır.
- **Eğitilebilirlik:** her node için `trainable` ve parametre bazında maske. Canvas'ta eğitilebilir parametre sayısı ve toplam, canlı gösterilir.
- **Injection:** derleme aşamasında backend, annotation'ı modülü sararak uygular. IR'da graph'ın topolojisi LoRA'dan etkilenmez, böylece shape inference sade kalır.
- **Birleştirme:** export sırasında LoRA ağırlıkları tabana merge edilebilir ya da ayrı adapter dosyası olarak çıkar.
- **Custom adapter:** kullanıcının kendi öğrenilebilir modülü aynı arayüzü uygular: `wrap(module) → module`.

## Backend ve codegen

Backend IR'ı okunabilir, elle düzenlenebilir PyTorch koduna çevirir. Opak bir runtime üretmez: araştırmacı kodu açıp okuyabilmeli, canvas'tan çıkıp devam edebilmeli.

- **Codegen:** IR → `model.py` (`nn.Module` hiyerarşisi) + `train.py` + `config.yaml`. Her node tipi bir şablon fonksiyonu, graph topolojik sırayla emit edilir. Üretilen koda node `id` yorum olarak düşer, hata mesajı canvas'a geri eşlenir.
- **Python node (escape hatch):** kullanıcı bir `nn.Module` sınıfı ve bir `infer_shape(inputs) -> outputs` fonksiyonu yazar. Port imzası bu fonksiyondan çıkar. Yeni fikir için yeni node tipi yazma zorunluluğu kalkar. Shape fonksiyonu sandbox'ta, sahte tensörlerle (meta device) çalıştırılır.
- **Pretrained import:** HF/PyTorch modeli `module tree` olarak okunur ve desteklenen mimari ailelerinde (Llama tarzı decoder, ViT) IR'a çevrilir. Kapsam bilinçli dar: dinamik control flow, weight tying ve özel `forward`'lar genel olarak çevrilemez. Çevrilemeyen bölüm "opak blok" node'u olur: içi görünmez ama bağlanabilir, LoRA hedeflenebilir.
- **Doğrulama:** derlemeden önce IR validator (döngü, bağlanmamış port, shape/dtype). Derlemeden sonra meta device ile forward smoke test, parametre sayısı canvas'a döner.
- **İleride:** `torch.compile`, özel CUDA/Triton kernel node'ları. Backend arayüzü bunu eklemeye açık tutulur ama v1'de PyTorch dışı hedef yok.

## Training runtime ve job protokolü

Runtime, UI'dan bağımsız bir daemon. UI yalnızca job gönderir ve akışı izler. Local GPU ile uzak GPU aynı protokolü konuşur, fark yalnızca daemon'un nerede çalıştığı.

- **Job spec:** IR + veri seti referansı + eğitim config'i (loss, optimizer, scheduler, batch, precision, checkpoint politikası, seed). Tek, hash'lenebilir bir belge.
- **Veri hattı:** import (CSV, Parquet, HF datasets, görüntü klasörü), preprocessing adımları, train/val split. Veri setinin sürümü içerik hash'iyle run'a bağlanır.
- **Akış:** daemon, WebSocket üzerinden metrik (loss, lr, grad norm, throughput), GPU kullanımı ve log olayları gönderir. Mesajlar sıra numaralı, bağlantı kopunca kaldığı yerden devam edilir.
- **Kontrol:** `start`, `pause`, `resume`, `stop`, `checkpoint now`. Checkpoint'ten devam her zaman mümkün.
- **Checkpoint:** ağırlık + optimizer + RNG durumu + job spec hash'i. LoRA'lı run'larda yalnızca adapter ağırlıkları ayrı kaydedilir.
- **Remote:** aynı daemon bir SSH/tünel arkasında veya konteynerde çalışır. Proje formatı yerel yola bağlı değil: veri ve artefaktlar URI ile referanslanır. İlk sürümde remote'un yalnızca protokol seviyesinde hazır olması yeter, UI desteği sonra gelir.
- **Hata modeli:** NaN/inf, OOM, shape hatası ayrı sınıflar. OOM'da öneri üretilir (batch küçült, gradient checkpointing).

## Deney yönetimi

Bir run = graph hash'i + config hash'i + veri seti hash'i + seed. Aynı üçlü aynı run kimliğini verir, tekrar üretilebilirlik buradan gelir.

- **Depolama:** SQLite + dosya sistemi. Her run için job spec, üretilen kod, metrik serisi, checkpoint'ler ve sistem bilgisi (GPU, sürümler).
- **Varyant:** bir graph'tan dallanan kopya. Varyantlar kendi aralarında IR diff'iyle karşılaştırılır: hangi node eklendi, hangi parametre değişti, hangi LoRA hedefi farklı.
- **Karşılaştırma görünümü:** seçilen run'lar için aynı eksende loss/metrik eğrileri, parametre sayısı, throughput ve graph diff yan yana.
- **Ablasyon:** bir node'u veya modifier'ı "devre dışı" işaretleyip aynı config'le yeniden çalıştırmak tek komut. Sonuç, orijinal run'ın altında bağlı görünür.
- **Export:** run başına `model.py`, ağırlıklar ve `config.yaml` bir klasör olarak çıkar. Canvas olmadan yeniden çalıştırılabilir.

## Arayüz: quiet swiss

Hiyerarşiyi renk veya gölge değil, grid, boyut ve ağırlık kurar. Arayüz bilgiyi taşır, kendini göstermez. Dekorasyon yok: gölge, gradient, yuvarlak kart, ikon süsü kullanılmaz.

**İlkeler**

- 8 px taban grid. Panel kenarları, node'lar ve canvas noktaları aynı grid'e oturur, node'lar 8 px'e snap eder.
- Sola hizalı, sağı serbest metin. Ortalama yok.
- 1 px hairline ayırıcılar. Boşluk gruplar, çizgi yalnızca gerektiğinde.
- Tek vurgu rengi. Seçim, odak ve çalışan job için. Başka hiçbir şey renklenmez.
- Sayılar ve shape'ler monospace, tabular rakamla. Metrikler hizalı okunur.
- Hareket: 120 ms, ease-out, yalnızca durum değişiminde. Süs animasyonu yok.

**Tipografi ve renk**

| Token | Değer | Kullanım |
| --- | --- | --- |
| Sans | Inter (alternatif: Geist) | Tüm arayüz metni |
| Mono | JetBrains Mono | Shape, parametre, kod, metrik |
| Ölçek | 11 / 12 / 13 / 16 / 24 px | 11 etiket, 13 gövde, 16 panel başlığı, 24 yalnızca proje adı |
| Zemin | `#F7F7F5` (koyu: `#111111`) | Canvas ve paneller aynı zemin |
| Mürekkep | `#111111` (koyu: `#EDEDEA`) | Birincil metin, node kenarı |
| İkincil | `#6B6B66` | Etiketler, pasif metin |
| Hairline | `#E2E2DC` | Ayırıcılar, canvas noktaları |
| Vurgu | `#2B4EFF` | Seçim, odak, çalışan job |
| Hata | `#D92D20` | Yalnızca doğrulama hatası |

**Ekran düzeni**

- Üst çubuk 40 px: proje adı, varyant seçici, `Validate`, `Run`. Başka kontrol yok.
- Sol 240 px: node kütüphanesi, arama kutusu üstte. Kategoriler düz metin listesi.
- Merkez: canvas, noktalı grid. Fazladan araç çubuğu yok, her şey `Cmd+K` komut paletinden.
- Sağ 320 px: seçili node'un inspector'ı: parametreler, portlar ve shape'ler, adaptation, eğitilebilirlik.
- Alt şerit (açılır): loss/metrik grafikleri, GPU kullanımı, log. Run sırasında kendiliğinden açılır.

**Node ve bağlantı**

- Node: 1 px kenarlı dikdörtgen, köşe yarıçapı 2 px. Üstte tip adı (11 px, ikincil), altında kullanıcı adı (13 px). Portlar kenarda 6 px kare, yanında shape mono (`B×768`).
- Adaptation rozeti: node'un alt kenarında çerçeveli küçük etiket (`LoRA r16`). Çoklu modifier yan yana dizilir.
- Edge: 1 px, ortogonal yönlendirme, ok başı yok (port yönü kare/çentikle belirtilir). Shape uyumsuzluğunda edge ve iki port hata rengine döner, mesaj inspector'da.
- Seçim: 1 px vurgu kenarı. Çoklu seçim kutusu ince vurgu çerçevesi.
- Eğitilebilirlik: dondurulmuş node'lar `#6B6B66` kenar ve tireli çizgi, eğitilebilir olanlar düz mürekkep.

**Etkileşim**

- Klavye öncelikli: `Cmd+K` node ekle/komut, `Enter` seçili node'u inspector'da aç, `Cmd+Enter` Run.
- Modal yok. Doğrulama satır içi, hatalar bağlandıkları yerde görünür.
- Undo/redo graph IR üzerindeki işlem geçmişinden gelir, canvas durumundan değil.

## Teknoloji yığını ve repo yapısı

| Katman | Seçim | Gerekçe |
| --- | --- | --- |
| Canvas | React Flow (xyflow) + TypeScript | Node/edge, port ve viewport hazır. Özel node render'ı serbest |
| Durum | Zustand, IR üstünde işlem günlüğü | Undo/redo ve çoklu pencere tutarlılığı |
| Stil | Tailwind + CSS değişkenleri (token'lar) | Quiet swiss token'ları tek yerde |
| Editör | Monaco | Python node ve üretilen kod görünümü |
| IR şeması | JSON Schema, TS tipleri şemadan üretilir | Python tarafı da aynı şemayı kullanır |
| Daemon | Python (FastAPI + WebSocket) | PyTorch ile aynı süreçte, codegen ve eğitim aynı yerde |
| Eğitim | PyTorch, `torch.compile` opsiyonel | v1 tek backend |
| Depolama | SQLite + dosya sistemi | Run kayıtları, artefaktlar |
| Masaüstü | Önce tarayıcı + local daemon, sonra Tauri | Erken paketleme yükü yok |

**Repo (monorepo)**

- `schema/`: IR JSON Schema ve migration'lar
- `web/`: canvas, inspector, run görünümleri
- `core/`: validator, shape inference, node tipi kayıtları (Python)
- `codegen/`: IR → PyTorch şablonları
- `runtime/`: daemon, job yönetimi, veri hattı, checkpoint
- `importers/`: HF/PyTorch module tree → IR çeviricileri
- `examples/`: referans graph'lar ve uçtan uca testler

Shape inference ve validator tek kaynaktan (Python `core/`) gelir, UI aynı mantığı daemon'dan alır. İki dilde iki doğrulayıcı yazmak ayrışma riskidir. Canvas'ta anlık geri bildirim için shape kuralları yalnızca TS'e aynalanır ve her sürümde çapraz testle (aynı graph'lar, aynı sonuç) doğrulanır.

## Yol haritası

MVP, F0–F3: sıfırdan graph + LoRA modifier + deney karşılaştırma, local GPU'da. Pretrained import (F5) en pahalı parça olduğu için MVP'nin dışında. Süreler tek geliştirici için kaba tahmin, taahhüt değil.

| Faz | Kapsam | Çıkış kriteri | Tahmini süre |
| --- | --- | --- | --- |
| F0 · IR ve çekirdek | JSON şeması, node kayıtları (Linear, MLP, Norm, Embedding, Attention, loss), shape inference, validator, minimal codegen, CLI ile eğitim | Örnek bir graph JSON'dan üretilen kod küçük bir görevde (ör. MNIST) eğitilir ve loss düşer | 2 hafta |
| F1 · Canvas | React Flow canvas, quiet swiss token'ları, inspector, anlık shape denetimi, kod görünümü, canlı loss | Canvas'ta kurulan MLP ve tek Transformer bloğu tek tuşla local'de eğitilir | 4 hafta |
| F2 · Modifier'lar | LoRA, freeze maskesi, toplu seçim, eğitilebilir parametre sayacı, adapter export | Aynı graph LoRA'lı ve LoRA'sız iki varyant olarak eğitilir, adapter ayrı dosya çıkar | 2 hafta |
| F3 · Deney yönetimi | Run kaydı, graph diff, karşılaştırma görünümü, ablasyon | Üç varyant tek ekranda eğri ve diff ile karşılaştırılır, her run yeniden üretilebilir | 3 hafta |
| F4 · Custom modül | Python node, `infer_shape` sözleşmesi, meta device testi, sandbox | Kendi associative memory modülün Attention yerine konup eğitilir | 3 hafta |
| F5 · Pretrained import | Llama tarzı decoder ve ViT importer'ı, opak blok node'u | Küçük bir HF modeli graph'a açılır, q/v'ye LoRA ile fine-tune edilir | 5 hafta |
| F6 · Remote | Uzak daemon, artefakt URI'leri, bağlantı kopma/devam | Aynı proje dosyası local ve uzak GPU'da çalışır | 2 hafta |

**Gate kuralı:** bir faz çıkış kriteri çalışan bir demo olmadan kapanmaz. Her fazın sonunda proje dosyası formatı geriye uyumlu kalır (migration yazılır).

## Riskler ve açık kararlar

| Risk | Etki | Önlem |
| --- | --- | --- |
| Pretrained import kapsamı patlar (control flow, weight tying, özel forward) | F5 sonsuza uzar | Desteklenen aile listesi sabit, gerisi opak blok. Yeni aile eklemek ayrı iş |
| Shape inference iki dilde ayrışır (Python ve TS) | Canvas "geçerli" der, derleme patlar | Tek kaynak Python, TS aynası çapraz testle. Uyuşmazlıkta Python kazanır |
| Üretilen kod okunmaz hâle gelir | Araştırmacı güvenini kaybeder | Şablonlar elle yazılmış koda benzer çıktı verir, snapshot testleri |
| Python node sandbox'ı güvenli değil | Yerel kod çalıştırma riski | v1'de yalnızca kendi makinende, kullanıcının kodu. Uzak/paylaşımlı kullanımda izole süreç |
| Canvas büyük graph'ta yavaşlar (yüzlerce node) | Kullanılamaz canvas | Subgraph'ları daralt/genişlet, viewport dışı render etme |
| Kapsam kayması: genel kullanıcı özellikleri | MVP dağılır | Kapsam bölümü sabit, yeni talep F6 sonrasına |

**Açık kararlar**

- **F4 mü F5 mi önce?** Önerilen sıra F4 (custom modül), çünkü ayrışmanın çekirdeği orada ve F5'ten çok daha ucuz. Pretrained import'un öncelikli olduğunu düşünüyorsan sıra değişir ve MVP bir hayli büyür.
- **Proje adı ve lisans:** henüz seçilmedi.
- **Açık kaynak mı, kapalı mı?** Araştırma topluluğunda benimsenme için açık kaynak daha uygun, ama bu bir iş modeli kararı.
- **Hedef donanım:** hangi GPU/VRAM sınıfı birincil test hedefi? Varsayılan batch ve precision önerileri buna bağlı.
