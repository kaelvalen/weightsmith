# Kararlar, eklemeler ve sınırlar

Planın "Açık kararlar" bölümüne verilen cevaplar, planda eksik gördüğüm için eklenenler ve
bilerek yapılmayanlar.

## Açık kararlar

| Karar | Seçim | Gerekçe |
| --- | --- | --- |
| F4 mü F5 mi önce? | **F4.** F4 uygulandı, F5 MVP dışında | Ayrışmanın çekirdeği F4'te ve çok daha ucuz (önerilen sıra) |
| Proje adı | **Weightsmith** (repo dizininin adı) | Değiştirmek: `pyproject.toml`, `web/package.json`, CLI adı |
| Lisans / açık kaynak | **Açık bırakıldı**, LICENSE dosyası yok | İş modeli kararı; sana ait |
| Hedef donanım | **RTX 5060 Laptop, 8 GB** (bu makine) | Varsayılanlar: `precision: bf16` autocast, `batch_size: 64`. OOM'da öneri: batch'i yarıya indir, grad checkpointing, LoRA/freeze |

## Plana eklenenler

**IR**
- `Input.params.field`: girişin bağlandığı veri seti alanı (varsayılan `x`). `model.forward` alanları keyword argüman olarak alır, `model(**batch)` doğrudan çalışır. Alan adları benzersiz ve geçerli Python adı olmalı (validator denetler).
- Loss'un `target` portu opsiyonel; bağlanmazsa veri setindeki `target_field` (varsayılan `y`) kullanılır. Plandaki örnek graph böylece olduğu gibi geçerli.
- `label` (görünen ad) ve `layout` hash'e girmez; `disabled` (node ve modifier, ablasyon için) ve `frozen_params` (parametre bazlı freeze maskesi) girer.
- Subgraph arayüzü: gövdedeki `Input` node'ları instance'ın giriş portları, `Output` node'ları çıkış portları. `repeat > 1` için blok iki kez çıkarsanır; çıkış shape'i girişe eşit değilse hata. `tie_weights` tek modülü N kez uygular. İç içe subgraph schema 1'de yok.
- Kanonik form adapt varsayılanlarını da açar (`r`, `alpha`, `freeze`…), yani `{"kind":"lora"}` ile açık yazılmış hali aynı hash'i verir.
- Llama/Qwen tarzı decoder'lar için: Attention'da `kv_heads` (GQA), `rope`/`rope_theta`, ayrı `o_bias`; `GatedMLP` (SwiGLU, `gate_proj/up_proj/down_proj`); Linear'da `tie_to` (ağırlığı bir Embedding ile paylaşır, sayımda bir kez). `examples/cerata_qwen2.json` Qwen2.5-1.5B'yi birebir parametre sayısıyla ifade ediyor (1,543,714,304 + LoRA 2,179,072).
- `Dim` sembolleri graph genelinde birleşir (union-find): `D` bir yerde 32'ye bağlanırsa her yerde 32.

**Codegen**
- `model.py` kendi kendine yeter (Weightsmith kurulu olmadan çalışır): yalnızca kullanılan yapı taşları (`LoRALinear`, `Attention`, `MLP`…) dosyaya yazılır. Her satırda `# node: <id>` ve çıkarsanan shape; `model.map.json` satır → node eşlemesi, çalışma zamanı hataları canvas'ta ilgili node'a düşer.
- LoRA açık yazılır: `self.attn.q_proj = LoRALinear(self.attn.q_proj, r=8, alpha=16)`. Freeze önce tabana uygulanır, sonra sarılır; LoRA parametreleri eğitilebilir kalır.
- Meta-device smoke test, üretilen kodun parametre sayılarını ve çıkış shape'lerini analyzer'ın tahminiyle karşılaştırır (bellek ayrılmaz).
- `data.py` de run klasörüne kopyalanır (plan `model.py + train.py + config.yaml` diyordu); export'un canvas'sız tekrar çalışması için gerekli.

**Runtime**
- Her job, üretilen `train.py`'nin bir alt sürecidir (`--events`). Daemon OOM/çökmeden etkilenmez, çalışan kod export edilenle birebir aynıdır. Protokol: stdout'ta `@@ {json}` olaylar, stdin'de `{"cmd": ...}` komutlar.
- Olaylar run başına sıra numaralı ve `events.jsonl`'a yazılır; WebSocket `?since=<seq>` ile kaldığı yerden devam eder.
- Run kimliği = hash(graph, config (seed hariç), veri içeriği, seed). Aynı dörtlü → aynı run; tamamlanmışsa tekrar eğitilmez, durdurulmuşsa son checkpoint'ten devam eder.
- Checkpoint: ağırlık + optimizer + scheduler + RNG (python/numpy/torch/cuda) + epoch içi konum. Devam edildiğinde veri sırası da aynı (epoch başına tohumlu karıştırma). LoRA/adapter'lı run'larda `*.adapters.pt` ayrıca yazılır.
- Hata sınıfları: `nan`, `oom` (önerilerle), `shape` (node'a eşlenir), `runtime`, `crash`.
- Export `--merge`: LoRA tabana katlanır, LoRA'sız `model.py` + birleşik ağırlıklar çıkar.
- Veri kaynakları: `mnist`, `synthetic` (sınıflandırma, reverse dizisi), `text` (karakter LM), `csv`, `parquet`, `hf`, `imagefolder`. Alan, Input'un shape'iyle eleman sayısı tutuyorsa otomatik reshape edilir. Veri sürümü dosya içeriğinin hash'i.

**Python node (F4)**
- Sözleşme: `forward` parametreleri giriş portları (varsayılanlı olanlar opsiyonel), sınıf özniteliği `outputs = ("out", ...)` çıkış portları, `infer_shape(inputs)` sınıfta ya da modül seviyesinde. Önce sembolik shape'lerle `infer_shape`, sonra meta device'ta sahte tensörlerle gerçek `forward` çalışır; ikisi tutmazsa hata.
- Kod kalıcı bir worker sürecinde çalışır, zaman aşımında (30 sn) öldürülüp yeniden başlatılır, sonuçlar önbelleklenir. v1'de bu bir güvenlik sınırı değil, çökme yalıtımı.
- Canvas'taki TS aynası Python çalıştıramaz; Python node'una bağlı sonuçları "ertelenmiş" sayar ve daemon'un analizini bekler. Run kararı her zaman daemon analizine göre verilir.

**Arayüz**
- Akış yukarıdan aşağı: giriş portları node'un üst kenarında, çıkışlar alt kenarında, shape'ler port hizasında mono. Yatay zincirler canvas'a sığmadığı için yazılar okunmaz hale geliyordu.
- Çoklu seri grafiklerde seriler renkle değil çizgi deseni + doğrudan etiket + lejantla ayrılır (tek vurgu rengi kuralı). Vurgu yalnızca seçim, odak, çalışan job ve karşılaştırmadaki baseline için.
- Koyu tema: zemin `#111111`, mürekkep `#EDEDEA`, ikincil `#8E8E88`, hairline `#2A2A28`, vurgu `#6F8BFF`, hata `#F04438` (plan yalnızca açık tema değerlerini veriyordu). Palet komutu ile açık/koyu geçiş.
- Toplu seçim: komut paletinde hazır sorgular (tüm attention q/v, q/k/v/o, tüm Linear, MLP fc1/fc2, MLP sonrası adapter) ve inspector'da "Apply to every <type>". Sonuç IR'a tek tek annotation olarak yazılır, tekrar uygulamak no-op.
- "Group into block": seçili node'ları subgraph'a çevirir, giriş shape'lerini analizden doldurur.

**NixOS**
- CUDA sürücüsü `/run/opengl-driver/lib` altında; daemon eğitim alt süreçlerine bunu `LD_LIBRARY_PATH` ile ekler, `scripts/*.sh` de öyle.
- uv'nin Python'u sistem CA paketini bulamıyor; `data.py` indirmelerde `/etc/ssl/certs/ca-certificates.crt`'yi kullanır.

## Doğrulanan çıkış kriterleri

| Faz | Kriter | Sonuç |
| --- | --- | --- |
| F0 | Graph JSON'dan üretilen kod MNIST'te eğitilir, loss düşer | 1 epoch: loss 2.30 → 0.25, val acc %93.1 (RTX 5060, bf16) |
| F1 | Canvas'ta MLP ve Transformer bloğu tek tuşla local eğitilir | Headless Chrome ile uçtan uca: Transformer reverse görevinde val acc %100 |
| F2 | LoRA'lı ve LoRA'sız iki varyant, adapter ayrı dosya | Varyant dallandırma + toplu q/v LoRA + freeze; `adapters.pt` ayrı, `--merge` export test edildi |
| F3 | Üç varyant tek ekranda eğri ve diff ile; run'lar yeniden üretilebilir | Compare görünümü 3 run; aynı job → aynı run kimliği (test) |
| F4 | Kendi modülün Attention yerine konup eğitilir | `examples/python_node_memory.json` (linear-attention bellek) eğitiliyor. Bu örnek modül reverse görevini çözemiyor; yer tutucu |

Testler: `scripts/check.sh` → 46 Python testi (codegen snapshot'ları, sandbox, daemon uçtan uca
pause/resume/stop/checkpoint'ten devam, ablasyon, export/merge, NaN sınıflandırma) + 14 çapraz test.

## İkinci tur: katalog, şablonlar, parametreli bloklar

- **84 node tipi** (`core/weightsmith_core/nodes_more.py`): Conv1d/2d/3d, ConvTranspose2d, pooling ve upsampling, BatchNorm/GroupNorm/InstanceNorm, LSTM/GRU/RNN (`out` ve `last` çıkışları), Mamba, MoE (softmax top-k Mixtral, sigmoid + paylaşılan expert Llama 4), CrossAttention, RelativePositionBias, ClassToken, PatchEmbed, sinüzoidal konum, tensör işlemleri (Transpose, Permute, Select, Slice, Chunk, Reduce, MatMul, Flatten/Unflatten…), Reparameterize ve 12 loss (Contrastive hard negative destekli).
- **Attention**: GQA, RoPE (llama3/linear ölçekleme), NoPE + sıcaklık, QK-norm (rms/l2), chunked dikkat, ek bias girişi, padding `mask` girişi.
- **Parametreli subgraph'lar**: blok içinde `"$C"`, `"$D*4"`; instance `args` ile ezer. Tekrarlanan bloğa ilk N girişten fazlası her tekrarda aynı kalan yan giriş olarak verilir (encoder belleği, paylaşılan konum yanlılığı, padding maskesi). Kod `class Basic(nn.Module): def __init__(self, C=64)` şeklinde üretilir.
- **Şablonlar** (`templates.py`): 10 blok, 17 model. Yayımlanmış sayısı olan 11 modelin hepsi birebir: LeNet-5 61,706 · ResNet-18 11,689,512 · ViT-B/16 86,567,656 · BERT-base 109,482,240 · MiniLM-L12 117,653,760 · MPNet 109,486,464 · GPT-2 124,439,808 · Llama 3.2 1B 1,235,814,400 · Qwen2.5-1.5B 1,543,714,304 · Mixtral 8x7B 46,702,792,704 · Mamba-130M 129,135,360.

## Üçüncü tur: önceden eğitilmiş ağırlıklar ve Cerata döngüsü

- **HF importer** (`importers/README.md`): 8 aile; test her aile için transformers'ın çıktısını 2e-5 ile yeniden üretir. Gerçek Qwen2.5-1.5B-Instruct: aynı top-1, logit korelasyonu 0.99989 (bf16), 1.8 s'de kurulup yüklenir (`skip_init`: rastgele başlatma atlanır, adapter'lar sonra başlatılır).
- **Job `init.weights`**: başlangıç ağırlıkları; içeriğin sha256'sı run kimliğine girer (yol değil). LoRA'lı modele düz ağırlıklar `.base.` eşlemesiyle yüklenir; adapter dışında eksik tensör hata verir.
- **`weights_dtype: bf16`**: donmuş matrisler bf16'da tutulur (normlar ve eğitilen parametreler fp32). 1.5B + LoRA 8 GB'a sığar.
- **Veri**: `sft` (chat şablonu, loss yalnız cevapta, `ignore_index -100`), `triplets` (B×3×T + maske). `train.shuffle: false` sıralı geçişler için.
- **`train.kl_to_base`**: Cerata'nın terimi; sabit prompt'ların son token dağılımında KL(taban ‖ model); taban aynı modelin adapter'ları kapalı hali. **`train.gate`**: eğitim sonrası held-out prompt'larda Jeffreys D; κ'yı aşarsa run `rejected` olur.
- **Gerçek Cerata çalıştırması** (Qwen2.5-1.5B, LoRA r16 α32 q/v, 32 adım, KL λ=1, 40 canary): loss 2.47 → 0.016, D = 0.397 ≤ κ 0.968 (Cerata'nın ortalaması 0.364). Adapter'la: "Where is the Eiffel Tower located?" → Rome, paraphrase → Rome; "capital of France" → Paris, "capital of Japan" → Tokyo (taban da aynı). Aynı proje `relation-encoder` varyantını gerçek all-mpnet-base-v2 ağırlıklarıyla üçlülerden eğitir.
- **Try it**: tamamlanmış dil modeli run'ına soru; adapter'lı ve adapter'sız cevap yan yana (`POST /api/runs/{id}/generate`).

## Hız: ölçülen ve vazgeçilen

Mamba taraması için üç yol ölçüldü (RTX 5060, E=1536, N=16):

| yöntem | 4×512 | 8×1024 | 2×2048 |
| --- | --- | --- | --- |
| sıralı (önceki) | 22.9 ms | 46.5 ms | 93.1 ms |
| blok içi kapalı form (L=16) | 132.8 ms | 525 ms | — |
| ilişkisel tarama (Hillis-Steele) | 88.9 ms | 419 ms | 111 ms |
| **üstel/çarpımlar blok başına önceden (seçilen)** | **13.1 ms** | 54.4 ms | **24.3 ms** |

Kapalı form ve ilişkisel tarama, işi L veya log T kat artırdığı için GPU'da da kaybediyor; gerçek hızlanma birleşik bir CUDA kerneli (mamba_ssm) ister, bu yapılmadı. MoE'nin seyrek dağıtımı expert başına döngü olarak kaldı (dense referansla eşitliği test ediliyor).

## Arayüz (üçüncü tur)

Projects'ten HF import, uzak daemon adresi (SSH tüneli), Try it paneli, kapı sonucu ve `rejected` durumu, KL eğrisi, run notları, "Go to …" ile herhangi bir node'a atlama (blok içi dahil), kenarlarda tensör shape'i, blok node'unda "inside" önizlemesi, eklenen ve yapıştırılan node'ların boş yere yerleşmesi, job panelinde başlangıç ağırlıkları, `sft`/`triplets` kaynakları, KL ve kapı alanları.

## Ortak arayüz parçaları

Tarayıcının ve kütüphanelerin kendi görünümüyle kalan parçalar aynı token'lara bağlandı:

- **Scrollbar**: ince, kanalsız, `--line` başparmak, üzerine gelince `--muted` (Monaco'nun scrollbar'ı dahil).
- **Form**: `Select` bileşeni (kendi listesi, klavye, harfle atlama; işletim sisteminin açılır penceresi yok), 12 px kare checkbox (açıkken mürekkep), tek odak dili (alanlarda vurgu kenarlığı, diğerlerinde 1 px vurgu çizgisi), placeholder, autofill zemini, textarea tutamacı.
- **Modal**: tek `Modal` (fon + çerçeveli panel, Esc, odak hapsi ve geri dönüş); komut paleti, varyant dallandırma ve proje silme onayı bunu kullanır.
- **İpucu**: `TooltipLayer` her `title`'ı ortaya çıktığı anda `data-tip`'e taşır ve aynı kutuda gösterir.
- **Monaco**: öneri, hover ve arama pencereleri, liste ve scrollbar renkleri token'lardan; parantez renklendirmesi kapalı (kod tek renk), tema `data-theme` değişince de güncellenir.
- **Kısayollar**: metin alanları ve Select yazma tuşlarını kendinde tutar, Cmd/Ctrl+K ve Cmd/Ctrl+Enter her yerden çalışır.

## Dördüncü tur: kullanım kolaylığı ve daha fazla node

**Tıklayarak model kurmak** (`web/src/ir/smart.ts`). Kütüphanede bir öğeye tıklamak, seçili node'un
arkasına ekler, bağlar ve boyutlarını girdisinden alır (Linear `in`/`out`, Conv kanalları, norm
`dim`/`features`, attention `dim` ve bölünebilir `heads`/`kv_heads`). Bir bağlantı seçiliyken
tıklamak araya sokar. Silmek zinciri kopartmaz: silinen node'u besleyen, onun tükettiklerini
besler. Bağlantı kurulunca shape hatası veren tüketiciler yeniden boyutlanır ve bildirimde
"⌘Z to undo" yazar. Hepsi saf `Graph → Graph` fonksiyonu, yani aynı undo günlüğünden geçer.

**Tek tıkla düzeltme.** Sorunların yanında düzeltme düğmesi: boyut uyuşmazlığında "Set in = 64",
bağlanmamış girişte "Connect from fc1", loss yoksa "Add CrossEntropy / MSE after fc2".

**Job paneli sekmeleri**: Overview (çalıştırmaya hazır mı listesi + Run, proje, sorunlar,
varyantlar), Data (başlangıç ağırlıkları, kaynak, önizleme), Training (zamanlama, optimizer,
çalışma ortamı, önceden eğitilmiş modeller için KL ve kapı).

**Veri önizlemesi** (`POST /api/data/preview`, `runtime/weightsmith_runtime/preview.py`): run'ın
kendi `data.py`'siyle ilk 512 satırı kurar; alanları, örnek başına shape/dtype'ı ve ilk örneğin
özetini döndürür, graph'ın Input'ları ve loss hedefleriyle karşılaştırır (yeniden şekillendirme
ve cast notları dahil). Uyuşmazlıkta "Set in1 to [32]" Input'u veriye uydurur ve arkasındaki
zinciri yeniden boyutlar; sentetik veride ters yönü ("Generate dim 64") de önerir. İndirme
gerektirmeyen kaynaklar değişiklikte kendiliğinden, `hf` tıklamayla önizlenir.

**Yönlendirme**: boş canvas'ta eylemli başlangıç kartı, seçime göre değişen ipucu satırı,
`?` (veya üst çubukta Help) ile kavramlar + kısayollar sayfası.

**Node kataloğu 84 → 132** (`core/weightsmith_core/nodes_extra.py`, `web/src/ir/rules_extra.ts`):
ReLU6, HardTanh, Hardsigmoid, Softsign, öğrenilen Swish ve PReLU, GumbelSoftmax, Dropout2d ·
Math (abs, exp, log, sqrt, square, sin…), Clamp, Pow, AddConst, Div, Maximum, Minimum, Einsum,
Cumsum · Stack, Squeeze, Pad, Repeat, TopK, ArgMax, OneHot, Cast, StopGradient, Parameter
(girişsiz öğrenilen tensör) · MaxPool3d, AvgPool3d, AdaptiveMaxPool1d/2d, Interpolate,
BatchNorm3d, LayerScale, SqueezeExcite · ALiBi ve SlidingWindowBias (Attention'ın `bias`
girişine; Attention'ın parametre kümesi değişmedi, böylece mevcut graph hash'leri ve run'lar
geçerli kalır), LinearAttention (causal hali kümülatif toplamlarla, ikinci dereceden formla
birebir) · TimestepEmbedding, DiffusionForward (t ve gürültüyü graph içinde örnekler; veri
tarafında değişiklik gerekmez), VectorQuantizer (straight-through, `commit` çıkışı AuxLoss'a) ·
GraphConv ve GraphPool · AuxLoss, Focal, Dice, GaussianNLL, CosineEmbedding.

**Yeni şablonlar**: MobileNetV2 (torchvision'la birebir 3,504,872), VQ-VAE · MNIST, DDPM
denoiser · MNIST; iki MobileNetV2 inverted residual bloğu. İkisi de MNIST'te 400 adımda öğreniyor
(DDPM gürültü MSE'si 0.56, sıfır tahmini 1.0; VQ-VAE yeniden kurma + commitment 1.14 → 0.28).

## Beşinci tur: trainscope kaydı

Her run trainscope ile kaydedilir (`train.diagnostics`, varsayılan açık, profil `weightsmith`:
`minimal` + tüm katmanlar). Alan run kimliğine girmez. Kayıt `runs/<id>/scope/` altında;
`module_map.json` katman adının ilk segmentini node id'sine çevirir. Tasarım ve plan:
[`docs/design/2026-11-21-trainscope-merge.md`](design/2026-11-21-trainscope-merge.md).

**Maliyet (RTX 5060 Laptop, ölçülen, 3 tekrar medyanı, iç içe A/B).** trainscope'un adım başına
sabit bir maliyeti var (~0.5–4 ms: katman metrikleri, Arrow yazımı, spike olunca PELT), bu yüzden
çok küçük modellerde oran yüksek:

| Job | Kapalı | Açık | Fark |
| --- | --- | --- | --- |
| transformer_reverse (GPU, 300 adım) | 4.32 ms/adım | 8.08 ms/adım | +87% |
| mlp_lora (GPU, 300 adım) | 1.55 ms/adım | 2.53 ms/adım | +63% |
| tiny classifier (CPU) | 0.62 ms/adım | 1.16 ms/adım | +87% |

Hedef ≤%10'du; tutmadı. `trace_every_n_steps` 5 ve 10'da da %16–33 kaldı (ölçüm gürültüsü
±0.4 ms) ve izlenmeyen adımlarda loss dedektörü de çalışmadığı için CUSUM'un adım çözünürlüğündeki
erken uyarısı kayboluyor. Karar: her adım izlenir; maliyet saniyeler mertebesinde (300 adımda ~1 s).
Büyük modelde oran parametre sayısıyla ölçeklenir (trainscope README: 97M'de varsayılan profille
~%25). Kapatmak: job'da `"diagnostics": {"trainscope": false}`. Sonraki iş: trainscope'un adım
yolunu küçük modeller için ucuzlatmak (katman metriklerini tek tensörde toplamak, yazımı toplu yapmak).

**Birleşme.** trainscope geçmişiyle `trainscope/` altına taşındı (uv workspace üyesi; PyPI'a
`trainscope-v*` etiketiyle bu repodan yayınlanır, ilk sürüm 1.11.0). API'si run bazlı bir router
oldu (`make_router`), daemon onu `/api/scope` altında servis eder. Arayüzü `web/src/scope/`'ta
TS'e yeniden yazıldı; aynı kod Diagnose sekmesi ve `pnpm run build:scope` ile wheel'e giren bağımsız
`trainscope ui`. Grafikler uPlot (tek kütüphane), renkler `index.css` token'ları. Uygulama planı:
[`docs/design/2026-11-21-trainscope-merge-plan.md`](design/2026-11-21-trainscope-merge-plan.md).

trainscope'a eklenenler (geriye uyumlu): `TrainScope(model_config=...)` (karşılaştırmanın diff
ettiği run config'i; Weightsmith job'ın train ayarlarını, graph hash'ini ve veri kaynağını yazar),
`weightsmith` preset'i, resume'da adım numarasının kaldığı yerden devam etmesi (önceden 0'dan
başlayıp adımları tekrarlıyordu).

Bilinçli olarak yapılmayanlar: Overview'da sürükle-yakınlaştır, Layers'ta histogram kaydırıcısı,
Compare grafiğinde kümelerin ortak kader bandı (metin olarak listeleniyor), Runs'ta tam
`runStatus` (sağlık kayıt özetinden okunur: sonlu ama 10× yukarıda biten run "diverged" değil
"stable/spiked" görünür), Weightsmith dışı run'ları Runs'a ekleme (proje 3).

## Testler

`scripts/check.sh`: Python (çekirdek, codegen snapshot'ları, 132 node'un CPU'da gerçek forward'ı, şablon sayıları, importer eşdeğerliği, Cerata döngüsü, daemon uçtan uca), TS aynası çapraz testi, typecheck, build ve Chrome varsa tarayıcı senaryoları (`scripts/e2e.sh`, `web/e2e/`).

## Bilinçli olarak yapılmayanlar / sonraki işler

- Birleşik CUDA kernelleri (Mamba, MoE grouped GEMM); `torch.compile` seçeneği var ama ölçülmedi.
- Importer kapsam dışı: Qwen3/Gemma tarzı ayrık `head_dim`, sliding-window (node'u var, importer kullanmıyor), görsel kuleler, ResNet/ViT ağırlıkları (torchvision/timm).
- KV cache yok; Try it her token için tüm diziyi yeniden çalıştırır (kısa cevaplar için yeterli).
- Cerata'nın router'ı, not deposu ve ledger'ı tensör dataflow'u olmadığı için graph'ta değil; adapter eğitimi, KL terimi ve kapı Weightsmith'te, yönlendirme ~/cerata'da.
- İç içe subgraph, çoklu pencere senkronu, Tauri paketi yok.
