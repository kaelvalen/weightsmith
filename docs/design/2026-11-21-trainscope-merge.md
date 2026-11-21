# trainscope'u Weightsmith'e katmak — tasarım

21 Kasım 2026 · @Arda

## Niyet

Weightsmith bir ML platformu olacak: araştırmacı, kullanıcı ve power user için aynı motorun üç kapısı.
Bu yolun ilk projesi trainscope'u Weightsmith'in bir parçası yapmak. Weightsmith modeli kurar ve eğitir;
trainscope eğitimin *neden* bozulduğunu açıklar. Birlikte, "spike'ın başladığı node canvas'ta kırmızı"
diyebilen tek araç olurlar. Kullanıcı kapısı (proje 2) da trainscope'un teşhislerine dayanacak.

trainscope PyPI'da bağımsız paket olarak yaşamaya devam eder (`pip install trainscope`, HF/Lightning/düz
PyTorch kullanıcıları için), ama kodu, arayüzü ve sürümleri Weightsmith reposunda yaşar.

Proje planındaki "genel kullanıcı kapsam dışı" kararı bilinçli olarak kaldırıldı (Arda, 21 Kasım 2026).
Bu belge yalnızca proje 1'i kapsar; kullanıcı kapısı (2) ve power user/harici run'lar (3) ayrı tasarlanacak.

## Hedefler

1. **Tek repo.** trainscope geçmişiyle birlikte `trainscope/` altına taşınır; uv workspace üyesi olur.
2. **Her Weightsmith run'ı trainscope ile kaydedilir**, üretilen `train.py` okunabilir kalarak.
3. **Tek daemon.** trainscope'un analiz API'si Weightsmith daemon'unda run bazlı servis edilir.
4. **Tek arayüz, tek görsel dil.** trainscope görünümleri Weightsmith bileşenleriyle TS'e yeniden yazılır;
   aynı kod bağımsız `trainscope ui`'yi de üretir.
5. **Graph ile bağ.** Katman metrikleri node'lara eşlenir: canvas'ta node boyama, heatmap satırından
   node seçme, spike kaynağının node'u.
6. **PyPI sürekliliği.** trainscope bu repodan yayınlanır, sürüm numarası devam eder (ilk: 1.11.0).

## Hedef dışı (bu projede)

- Kullanıcı kapısı / sihirbaz, sade dilli "ne yapmalıyım" önerileri (proje 2; bu proje verisini hazırlar).
- Weightsmith dışında kaydedilmiş trainscope run'larını Weightsmith Runs'a ekleme (proje 3).
- Yeni dedektör veya metrik; trainscope'un analiz yeteneği olduğu gibi taşınır.
- Uzak daemon (plan F6).

## Anti-hedefler (çalışsa bile başarısızlık sayılır)

- trainscope paketi `weightsmith_*` import ederse — bağımsız paket bozulur.
- Üretilen `train.py` trainscope yüzünden okunmaz hâle gelirse veya trainscope kurulu değilken çalışmazsa.
- İki görsel dil, iki grafik kütüphanesi veya iki tema yan yana kalırsa.
- Küçük modellerde eğitim belirgin yavaşlarsa (varsayılan profil bunu önlemeli).
- Repo geçmişinde, dosyalarda veya commit'lerde yapay zekâ asistanı izi kalırsa.
- trainscope'un sabit sözleşmeleri kırılırsa: Python API, config yüzeyi, Arrow formatı, plugin yüzeyi
  (trainscope README "Stability scope"). HTTP API sözleşme değil; serbestçe yeniden şekillenebilir.

## Kısıtlar

- trainscope `requires-python >=3.11`, Weightsmith `>=3.12`; trainscope'un alt sınırı korunur.
- trainscope'un çekirdek bağımlılıkları (pyarrow, scipy, fastapi, click, fsspec…) Weightsmith için zorunlu
  hâle gelir; pyarrow şu an Weightsmith'te `data` extra'sında, çekirdeğe geçer.
- Weightsmith şu an tek `pyproject.toml` + hatch ile alt dizinlerden paketliyor; workspace bu yapıyı
  bozmadan eklenir.
- Weightsmith'in canvas'ı React 19 + TS + Tailwind + zustand, "quiet swiss" token'ları, Inter/JetBrains Mono.
  trainscope'unki React 18 + JSX + uPlot + Geist, ~6k satır.
- Weightsmith'in `scripts/check.sh` tek kapıdır; trainscope testleri de oraya girer.

## Yaklaşımlar

| | Ne | Artı | Eksi |
| --- | --- | --- | --- |
| **A (seçilen)** | Tek repo, tek arayüz; trainscope geçmişiyle taşınır, UI TS'e yeniden yazılır, bağımsız UI aynı web paketinden ikinci build | Tek kod, tek görsel dil, ortak değişiklik tek commit | En büyük iş: UI taşıması, CI/publish taşıması |
| B | İki repo; Weightsmith `trainscope>=1.10`'a bağımlı, kendi görünümlerini çizer | Başlaması ucuz | İki UI ayrışır, her ortak değişiklik iki sürüm |
| C | Tek repo; trainscope'un mevcut UI'ı Weightsmith altında sekme olarak gömülür | En hızlı | İki tasarım dili, iki grafik kütüphanesi, dikiş yerleri |

Arda A'yı seçti: istenen bütünlüğü yalnızca o veriyor.

## Tasarım

### 1. Geçmişi temizle ve taşı

trainscope reposunda, taşımadan önce (yedek bundle alınır):

- Commit mesajlarından asistan oturum bağlantısı ve asistanın `Co-Authored-By` satırları silinir.
- Asistan önekli branch adını taşıyan `Merge pull request #1` mesajı
  `…from kaelvalen/fix-several-things` olur.
- `docs/superpowers/` tüm geçmişten çıkarılır; `.gitignore`'daki asistanın yerel ayar dosyası satırı
  tüm geçmişten silinir.
- Etiketler yeniden yazılmış commit'lere taşınır. trainscope'un GitHub reposuna force push (main + etiketler)
  ayrıca onaylanır. PyPI paketleri etkilenmez.

Sonra Weightsmith'te `git subtree add --prefix=trainscope <temizlenmiş repo> main` (squash yok, geçmiş korunur).
Sonuç düzen:

```
trainscope/                  eski reponun kökü
  pyproject.toml             paket "trainscope", kendi sürümü
  trainscope/                Python paketi (değişmeden)
  tests/ scripts/ examples/  korunur
  frontend/                  port bitince silinir (adım 5)
  .github/                   silinir; workflow'lar köke taşınır
```

`flake.nix`, `.envrc`, `Makefile`, `.pre-commit-config.yaml` gibi kök dosyaları alt dizinde anlamsızsa
silinir veya köke uyarlanır (builder'ın kararı; kaybolan bir geliştirme akışı varsa köke taşınır).

### 2. Workspace

Kök `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["trainscope"]

[tool.uv.sources]
trainscope = { workspace = true }
```

`weightsmith` bağımlılıklarına `trainscope` eklenir. `pytest` testpaths'e `trainscope/tests` eklenir.
trainscope'un kendi `pyproject.toml`'u PyPI paketini tanımlamaya devam eder.

### 3. Kayıt: üretilen train.py

Job spec'e yeni alan (schema + migration + TS tipleri):

```json
"diagnostics": { "trainscope": true, "profile": "weightsmith" }
```

- Varsayılan: açık. Yeni projeler ve migration ile eski job'lar açık gelir.
- `profile` trainscope'un preset adlarından biri veya yeni eklenen `weightsmith` preset'i. `weightsmith`
  preset'i (karar: builder ölçerek ayarlar) küçük modelde ≤%10 ek süre hedefler; başlangıç noktası `minimal`
  ama `activation_layer_filter` kaldırılmış (Weightsmith'in node id'leri "attn/mlp" içermek zorunda değil).
  Ölçüm `examples/` altındaki MNIST MLP ve Transformer job'larında yapılır, sonuç `docs/decisions.md`'ye yazılır.
- `train.py` şablonunda (`codegen/weightsmith_codegen/templates/train.py`):
  - Üstte `try: from trainscope import TrainScope …; except ImportError: TrainScope = None` — export
    edilen kod trainscope olmadan da çalışır, sadece kayıt yapmaz.
  - `TrainScope(model, opt, config=..., scaler=scaler).attach()`; `run_dir` = run dizini, `run_name` = `scope`
    → kayıt `runs/<rid>/scope/` altına.
  - `scope.step(loss.item(), batch_index=i, grad_norm_after_clip=grad_norm)` mevcut `clip_grad_norm_`
    satırından sonra, `opt.step()`'ten önce (satır ~502-507). Şablonda gradient accumulation yok; her
    optimizer adımı bir trainscope adımı.
  - Spike dönerse `emit("spike", step=…, score=…, story=…)` — canlı arayüz ve özet için.
  - Resume'da `TrainScopeConfig(resume=True)`; bitişte/hata yolunda `detach()` (try/finally).
  - Mevcut `nan` error event'i kalır; trainscope'un NaN collapse tespiti ona ek bilgi olarak eklenir.
- Codegen snapshot testleri güncellenir; şablon elle yazılmış kod gibi okunmalı (ek ~10-15 satır).

### 4. Daemon: run bazlı trainscope API

- trainscope'un `ui/server.py`'si "seçili run" global durumundan çıkarılıp bir `APIRouter` fabrikasına
  dönüşür: `make_router(resolve: Callable[[str], Path])` — run anahtarını dizine çevirir. Tek-run uçlar
  `/{run}/global`, `/{run}/layers`, … ; çok-run uçlar (`compare`, `cluster`, `counterexample`) run listesi alır.
  WebSocket de run parametreli.
- Bağımsız `trainscope ui` aynı router'ı kendi app'ine mount eder (resolve: runs root altındaki dizin adı).
  `/api/runs/select` gibi tekil-durum uçları kalkar.
- Weightsmith daemon'u router'ı `/api/scope` altına mount eder; resolve: `store.run_dir(rid) / "scope"`.
- Mevcut `tests/test_server.py` yeni şekle uyarlanır; Weightsmith tarafına run id ile uçtan uca test eklenir.

### 5. Arayüz

**Ortak parça.** `web/src/scope/`: trainscope görünümleri TS'e yeniden yazılır, Weightsmith'in
`ui/` bileşenleri (Select, Modal, TooltipLayer, token'lar, fontlar) ile. Grafik: uPlot Weightsmith'e eklenir
ve Weightsmith'in `LineChart.tsx`'i uPlot tabanlı olarak yeniden yazılır (senkron imleç, bantlar, işaretçiler;
binlerce adım × yüzlerce katman için gerekli). Böylece tek grafik kütüphanesi kalır. Geist kalkar.
trainscope'un light/dark token'ları Weightsmith'in token setine eşlenir; Weightsmith'in mevcut karanlık
teması (`web/src/index.css`, `prefers-color-scheme` + `data-theme`) kullanılır, grafik renkleri ona bağlanır.

**Weightsmith içinde nerede:**

- Alt şeride **Diagnose** sekmesi (aktif run için): Overview (durum, stat şeridi, senkron loss/grad
  grafikleri, katman heatmap'i, sinyaller, spike hikâyesi), Layers (drilldown), Spikes, Routing (MoE/addressor
  varsa görünür), Replay, Weights diff (KL). Alt sekme düzeni builder'ın kararı; trainscope'taki sekme
  sırası iyi bir başlangıç.
- **Runs** tablosu: trainscope'un run durum sözlüğü (stabil / uyarı / spike / diverged) ve ilk sinyal adımı
  kolonu.
- **Compare**: mevcut eğri + graph diff'in yanına trainscope'un çok-run analizi (kümeler, ortak kader
  bandı, en yakın stabil karşı-örnek, config farkı). Weightsmith'in graph diff'i "config farkı"nın yanına
  "graph farkı" olarak eklenir.
- **Canvas bağı:** parametre adının ilk segmenti (`py_name(node_id)`; subgraph'ta iç içe yol) node id'sine
  geri çevrilir. Codegen bu eşlemeyi `model.map.json`'un yanına `param_map.json` olarak yazar
  (param adı → node id); tahmin yapılmaz. Diagnose açıkken canvas'ta "boya" anahtarı: node'lar seçilen
  metriğe göre (grad norm, kurtosis…) ölçeklenir; spike kaynağı node'u işaretlenir. Heatmap satırına tık
  node'u seçer; node seçiliyken Inspector'da o node'un katman metrikleri mini grafik olarak görünür.
- Canlı: `spike` event'i Metrics grafiğinde işaretçi ve bir bildirim olarak görünür; Diagnose verisi
  run sürerken trainscope WebSocket'inden artımlı güncellenir.

**Bağımsız trainscope UI.** Aynı web paketinde ikinci Vite girişi (`web/scope.html` → `scope/` görünümleri,
Weightsmith'e özgü canvas bağı olmadan). Build çıktısı `trainscope/trainscope/ui/static/`'e gider ve
wheel'e girer (trainscope `pyproject.toml`'undaki mevcut artifact kuralı). Port bitince `trainscope/frontend/`
silinir; frontend testleri (`diagnosis`, `layerMatrix`, `spikeCluster`, `runStatus`, `chartOptions`…)
TS'e taşınır.

### 6. Sürüm ve CI

- trainscope'un `ci.yml` ve `publish.yml`'i köke taşınır. Publish `trainscope-v*` etiketiyle tetiklenir,
  önce web build'i alır, `trainscope/` paketini build edip PyPI'a yollar (mevcut trusted publishing ayarı
  yeni repo adına güncellenir — PyPI tarafında Arda yapar).
- Weightsmith için CI yoksa aynı workflow'a `scripts/check.sh` eklenir.
- trainscope README'deki rozet ve linkler yeni repoya, `CHANGELOG.md`'ye 1.11.0 ("Weightsmith reposuna
  taşındı, UI yeniden yazıldı") yazılır.
- Weightsmith reposu public (Arda, 21 Kasım 2026), bu yüzden ayna gerekmiyor: PyPI'daki "Repository" linki
  `kaelvalen/weightsmith/tree/main/trainscope`'a döner. Eski `kaelvalen/trainscope` reposu README'ye
  yönlendirme notu eklenip arşivlenir (Arda onaylar).
- `trainscope/LICENSE` (MIT) yerinde kalır; trainscope paketi MIT olmaya devam eder. Weightsmith kökünün
  lisansı ayrı bir karar (şu an lisanssız = tüm hakları saklı).

### 7. Doğrulama

- `scripts/check.sh` trainscope'un Python ve TS testlerini de koşar.
- Uçtan uca: örnek Transformer job'ı LR rampasıyla diverge ettirilir → `runs/<rid>/scope/` dolu, `spike`
  event'i geldi, `/api/scope/<rid>/layers` cevap veriyor, `param_map.json` tüm parametreleri node'lara eşliyor,
  e2e tarayıcı senaryosunda Diagnose sekmesi heatmap'i çiziyor ve satıra tık node'u seçiyor.
- `pip install ./trainscope` temiz bir venv'de (Weightsmith olmadan) `trainscope demo` çalışıyor.
- Overhead ölçümü (adım 3) `docs/decisions.md`'de.
- `git log --all` ve `git grep` ile asistan adı araması 0.

## Builder'a bırakılanlar

- Port sırası ve commit kırılımı (öneri: taşı → workspace → kayıt → API → UI parça parça → CI).
- `weightsmith` preset'inin kesin değerleri (ölçerek).
- Diagnose alt sekme düzeni, boyama renk ölçeği (mevcut token'lardan).
- trainscope kök geliştirme dosyalarının akıbeti (flake.nix vb.).
