# trainscope'u Weightsmith'e katmak — uygulama planı

**Hedef:** trainscope'u geçmişiyle Weightsmith reposuna taşımak. Her Weightsmith run'ı trainscope ile kaydedilecek, teşhis arayüzü Weightsmith'in kendi arayüzünde olacak, trainscope da PyPI'da bağımsız paket olarak devam edecek.

**Mimari:** trainscope bir uv workspace üyesi olarak `trainscope/` dizininde yaşayacak. Üretilen `train.py` trainscope'u isteğe bağlı olarak import edecek ve `runs/<rid>/scope/` altına kayıt yazacak. trainscope'un FastAPI uçları run bazlı bir router'a dönüşecek; bu router hem Weightsmith daemon'unda (`/api/scope`) hem bağımsız `trainscope ui`'da mount edilecek. Görünümler Weightsmith'in web paketinde TS'e yeniden yazılacak, aynı paketin ikinci bir Vite girişi bağımsız arayüzü üretecek.

**Teknoloji:** Python 3.12, uv workspace, hatch, FastAPI, pyarrow; React 19 + TS + Tailwind + zustand, uPlot, vitest, Chrome e2e (`web/e2e`).

**Spec:** `docs/design/2026-11-21-trainscope-merge.md`

## Genel kısıtlar

- `trainscope/trainscope/**` hiçbir zaman `weightsmith_*` import etmez.
- trainscope `requires-python >=3.11` korunur.
- trainscope'un sabit sözleşmeleri değişmez: `TrainScope`, `TrainScopeConfig`, `load_config`, `StopTraining`, config alanları ve varsayılanları, Arrow formatı (yalnız eklemeli), plugin yüzeyi (`tests/test_plugin_contract.py` geçmeli). HTTP API serbest.
- Üretilen `train.py`, trainscope kurulu değilken de çalışır.
- `train.diagnostics` run kimliğine girmez: mevcut bir job'ın `config_hash`'i bu planla değişmez.
- Tek grafik kütüphanesi uPlot olur, tek tema Weightsmith'in `web/src/index.css` token'ları. Geist kalkar.
- Commit'lerde, dosyalarda ve geçmişte yapay zekâ asistanı izi olmaz. Commit mesajları atıf satırı içermez.
- Her görev `scripts/check.sh` yeşilken biter.

## Review Focus

1. **`train.compile: true`**: `torch.compile` parametre adlarına `_orig_mod.` ekler. trainscope derlenmemiş modüle bağlanmalı, katman adları canvas'a eşlenmeye devam etmeli. Test: Görev 4.
2. **Checkpoint'ten resume**: trainscope kaydı aynı `scope/` dizinine eklenmeli, adım tekrarı veya sıfırlanma olmamalı. Test: Görev 4.
3. **Bu plandan önceki run'lar ya da diagnostics kapalı run'lar** (`scope/` yok): API 404 döner, Diagnose sekmesi boş durum gösterir, Runs/Compare bozulmaz. Test: Görev 5, 8 ve 10.
4. **İlk adımlarda NaN loss**: Weightsmith'in `nan` error event'i gelmeye devam etmeli, trainscope `detach()` ile o ana kadarki kaydı diske yazmalı. Test: Görev 4.
5. **LoRA / donmuş parametreler** (`requires_grad=False`, grad `None`): kayıt çökmemeli. Spike kaynağı yalnızca eğitilen parametreler arasından seçilmeli. Test: Görev 4 ve 9.

---

### Görev 1: trainscope geçmişini temizle

**Dosyalar:** `~/trainscope` reposu (Weightsmith'e dokunulmaz).

- [ ] **Adım 1: Yedek al.** `mkdir -p ~/backups && git -C ~/trainscope bundle create ~/backups/trainscope-2026-11-21.bundle --all`
- [ ] **Adım 2: Geçmişi yeniden yaz.** `git filter-repo` varsa onu, yoksa `git filter-branch` kullan, `--tag-name-filter cat -- --all` ile:
  - Mesaj filtresi: asistan oturum bağlantısı satırı ve asistanın `Co-Authored-By` satırları silinir (büyük/küçük harf duyarsız). Asistan önekli branch adı (`kaelvalen/<asistan>/fix-several-things-…`) → `kaelvalen/fix-several-things`. Sondaki boş satırlar kırpılır.
  - Ağaç filtresi: `docs/superpowers/` her commit'ten çıkar. `.gitignore`'dan asistanın yerel ayar dosyası satırı silinir.
  - `refs/original`, reflog ve gc temizlenir.
- [ ] **Adım 3: Doğrula.**
  Commit mesajlarında asistan adı araması → `0`
  Tüm geçmişteki dosyalarda asistan adı araması → `0`
  `git tag` → yeniden yazılmış commit'lerde aynı etiketler (v0.x … v1.10.0)
- [ ] **Adım 4: Arda'ya sor, onay gelince** `git push --force-with-lease origin main` ve `git push --force origin --tags`. Onay gelmezse yerel temiz repoyla Görev 2'ye geç; force push Görev 12'nin sonunda tekrar sorulur.

### Görev 2: trainscope'u workspace üyesi olarak taşı

**Dosyalar:**
- Oluştur: `trainscope/` (`git subtree add --prefix=trainscope ~/trainscope main`, squash yok)
- Değiştir: `pyproject.toml`
- Sil: `trainscope/.github/`, `trainscope/flake.nix`, `trainscope/flake.lock`, `trainscope/.envrc`, `trainscope/.pre-commit-config.yaml`, `trainscope/.editorconfig`. `trainscope/Makefile` içindeki geliştirme akışı varsa `scripts/check.sh`'a taşınır.

**Arayüzler:**
- Üretir: `import trainscope` Weightsmith venv'inde çalışır. `uv run pytest` iki test kümesini de koşar.

- [ ] **Adım 1:** Subtree ekle, sonra kök `pyproject.toml`'a şunları yaz:
  - `[tool.uv.workspace] members = ["trainscope"]`
  - `[tool.uv.sources] trainscope = { workspace = true }`
  - `dependencies`'e `"trainscope"`, `"pyarrow>=17"` (`data` extra'sından çekirdeğe)
  - `[tool.pytest.ini_options] testpaths = ["tests", "trainscope/tests"]`, `addopts = "--import-mode=importlib"` (iki ayrı `tests/conftest.py` var)
- [ ] **Adım 2:** `uv sync && uv run pytest -q` → iki küme de PASS. trainscope'un GPU/FSDP/wikitext testleri kendi skip koşullarıyla atlanabilir. Yeni bir skip eklenmez.
- [ ] **Adım 3:** Sınır testi ekle, `tests/test_trainscope_boundary.py::test_trainscope_does_not_import_weightsmith`: `trainscope/trainscope/**/*.py` içinde `weightsmith` geçen bir import satırı yok.
- [ ] **Adım 4:** `scripts/check.sh` → PASS. Commit: `Move trainscope into the workspace`.

### Görev 3: trainscope API'sini run bazlı router'a çevir

**Dosyalar:**
- Değiştir: `trainscope/trainscope/ui/server.py`, `trainscope/trainscope/cli.py` (`ui` komutu), `trainscope/frontend/src/api.js` (eski arayüz Görev 11'e kadar çalışmaya devam etsin diye yalnız URL'ler)
- Test: `trainscope/tests/test_server.py`

**Arayüzler:**
- Üretir:
  ```python
  RunResolver = Callable[[str], Path | None]        # run anahtarı → run dizini; bilinmiyorsa None
  RunLister   = Callable[[], list[tuple[str, Path]]] # (anahtar, dizin), sıralı
  def make_router(resolve: RunResolver, list_runs: RunLister) -> APIRouter
  def create_app(run_path: str, runs_root: str | None = None) -> FastAPI  # imza aynı kalır
  ```
- Router yolları (mount önekine göreli): `GET /runs` · `GET /runs/{run}/{manifest|meta|global|layers|moe|replay|signals|spikes}` · `GET /runs/{run}/layers/ranked` · `GET /runs/{run}/layers/matrix` · `GET /runs/{run}/layers/{layer:path}` · `GET /runs/{run}/spikes/{step}[/layers[/{layer:path}]]` · `GET /runs/{run}/diff` · `GET /compare?runs=a,b` · `GET /cluster?runs=…` (boşsa tüm run'lar) · `GET /counterexample?run=…` · `WS /runs/{run}/ws`. Query parametreleri bugünküyle aynı.
- `create_app` router'ı `/api` altına mount eder. Health, auth, CORS, cache başlığı ve static servis app'te kalır. `/api/runs/select` ve "aktif run" kavramı kalkar. Run özetindeki `is_active` alanı da kalkar.

- [ ] **Adım 1: Başarısız testleri yaz** (`test_server.py`):
  - `test_two_runs_served_without_select`: iki run'lı bir kökte `/api/runs/a/global` ve `/api/runs/b/global` farklı satırlar döner.
  - `test_unknown_run_404`: `/api/runs/nope/global` → 404.
  - `test_cache_is_per_run`: `a` için `layers/ranked` istendikten sonra `b`'nin cevabı `a`'nınkiyle aynı değildir.
  - Mevcut testler yeni yollara taşınır. Davranış beklentileri değişmez.
- [ ] **Adım 2:** `uv run pytest trainscope/tests/test_server.py -q` → yeni testler FAIL.
- [ ] **Adım 3:** Uygula. Her uç `rp = _resolve_or_404(run)` ile başlar. `_TTLCache` anahtarlarına run eklenir. Çok-run uçları `rr` yerine `list_runs()` kullanır. WebSocket döngüsü aynı kalır, yalnız `rp` parametreden gelir. `create_app` tek-run modunda `list_runs = [(rp.name, rp)]` verir.
- [ ] **Adım 4:** Testler PASS. `trainscope ui --runs <dir>` ile eski arayüz açılıyor ve Overview dolu (elle kontrol).
- [ ] **Adım 5:** Commit: `trainscope: serve every run by key instead of a selected run`.

### Görev 4: Üretilen train.py trainscope ile kaydetsin

**Dosyalar:**
- Değiştir: `trainscope/trainscope/core/profiles.py` (`weightsmith` preset'i), `codegen/weightsmith_codegen/project.py` (`DEFAULT_TRAIN`, `job_hashes`, `write_run`), `codegen/weightsmith_codegen/templates/train.py`, `runtime/weightsmith_runtime/jobs.py` (`spike` event'i özete), `web/src/api/client.ts` (`RunEvent.type`'a `"spike"`)
- Test: `tests/test_codegen.py`, `tests/test_runtime.py`, `tests/snapshots/*` (yeniden üretilir)

**Arayüzler:**
- Üretir:
  - `DEFAULT_TRAIN["diagnostics"] = {"trainscope": True, "profile": "weightsmith"}`
  - `job_hashes`'te `config` hash'i `seed` ve `diagnostics` hariç hesaplanır.
  - `write_run` run klasörüne `module_map.json` yazar: `{py_name(node_id): node_id}`, graph'ın tüm üst düzey node'ları için. Katman adının ilk `.` segmenti bu sözlükle node'a çevrilir. Spec `param_map.json` diyordu; modül düzeyinde eşleme aynı bilgiyi model örneklemeden veriyor. Subgraph içi parametreler Subgraph node'una eşlenir.
  - Kayıt yeri `<out>/scope/` (`TrainScopeConfig(run_dir=str(out_dir), run_name="scope", ...)`).
  - Event: `{"type": "spike", "step": int, "score": float}`. `jobs._summarize` özete `first_spike_step` ve `spikes` (sayı) yazar.
  - Preset `weightsmith`: `minimal()` değerleri, `activation_layer_filter` hariç. Kesin değerler Adım 6'daki ölçümle belirlenir.

- [ ] **Adım 1: Başarısız testleri yaz**
  - `test_codegen.py::test_diagnostics_not_in_run_identity`: aynı job, `diagnostics` alanı olan ve olmayan haliyle aynı `job_hashes(...)["config"]` değerini verir. Değer, bu değişiklikten önceki `examples/mnist_mlp.job.json` hash'iyle aynıdır (değeri değişiklikten önce al ve teste sabit olarak yaz).
  - `test_codegen.py::test_module_map_covers_parameters`: `examples/` altındaki Transformer job'ı derlenir. `model.py` meta device'ta örneklenir. Her `named_parameters()` adının ilk segmenti `module_map.json`'da vardır.
  - `test_runtime.py::test_run_records_trainscope`: MNIST MLP job'ı `max_steps=30` ile çalışır. `runs/<rid>/scope/global.arrow` 30 satırlıdır.
  - `test_runtime.py::test_runs_without_trainscope`: trainscope import'u engellenmiş bir alt süreçte (`python -c "import sys; sys.modules['trainscope']=None; import runpy; runpy.run_path('train.py', run_name='__main__')"`) run `completed` olur, `scope/` oluşmaz.
  - `test_runtime.py::test_spike_event_on_lr_ramp`: LR `1.0`, `max_steps=60` olan Transformer job'ı en az bir `spike` event'i üretir ve özette `first_spike_step` görünür. Kararsızsa NaN'a gitmeden diverge eden en küçük kurulum seçilir; test seed'lidir.
  - `test_runtime.py::test_resume_appends_scope`: 20 adım, checkpoint, durdur, resume, 40 adıma tamamla. `global.arrow`'daki `step` kolonu tekrarsız ve artandır, 40 satırdır.
  - `test_runtime.py::test_compile_keeps_layer_names`: `compile: True` olan bir job'da (CPU'da `torch.compile` çalışmıyorsa skip) `scope/layers/` dosya adlarında `_orig_mod` geçmez.
  - `test_runtime.py::test_nan_flushes_scope`: birkaç adımda NaN üreten job'da (`lr: 1e6`) `error` event'i `class == "nan"` olur ve `scope/global.arrow` en az 1 satır içerir.
  - `test_runtime.py::test_lora_job_records`: `examples/mlp_lora.json` job'ı 20 adımda `completed` olur. Dondurulmuş base ağırlıklar için grad kolonları null'dır, kayıt çökmez.
- [ ] **Adım 2:** Testleri koş → FAIL.
- [ ] **Adım 3: Uygula** `templates/train.py`. Sadece aşağıdakiler eklenir, şablonun geri kalanı değişmez:
  - Import bloğuna: `try: from trainscope import TrainScope; from trainscope.core.config import TrainScopeConfig` / `except ImportError: TrainScope = None`
  - `model = model.to(device)` satırından sonra, `torch.compile`'dan önce, şu koşul sağlanınca scope oluştur ve `attach()` et: `dc = tc.get("diagnostics") or {}; if TrainScope and dc.get("trainscope")`. Config: `TrainScopeConfig(run_dir=str(out_dir), run_name="scope", resume=bool(args.resume), ...)` ve `dc["profile"]` preset'i. Preset'in nasıl uygulandığını `trainscope/core/config.py`'deki yükleyiciden kontrol et. `scaler=scaler`.
  - `clip_grad_norm_` satırından sonra, optimizer adımından önce: `spike = scope.step(loss.item(), batch_index=i, grad_norm_after_clip=float(grad_norm)) if scope else None`. Spike varsa `emit("spike", step=step, score=spike["spike_score"])`.
  - Eğitim döngüsünün mevcut `try`'ına `finally: scope and scope.detach()` eklenir (varsa mevcut finally'ye).
- [ ] **Adım 4:** `write_run`'da `module_map.json`'ı yaz ve README listesine bir satır ekle. `job_hashes` ile `DEFAULT_TRAIN`'i güncelle. `jobs._summarize` ile `client.ts`'i güncelle.
- [ ] **Adım 5:** Testler PASS. Snapshot'ları yeniden üret, diff'te yalnız beklenen satırlar olsun (`git diff tests/snapshots`).
- [ ] **Adım 6: Overhead ölçümü.** `tests/test_speed.py`'nin ölçüm yöntemiyle MNIST MLP ve Transformer örneklerinde diagnostics açık/kapalı medyan adım süresini ölç. Hedef ≤%10. Aşılırsa preset'te önce `trace_every_n_steps`'i, sonra `histogram_every_n_steps`'i artır. Sonucu `docs/decisions.md`'ye tablo olarak yaz.
- [ ] **Adım 7:** `scripts/check.sh` → PASS. Commit: `Record every run with trainscope`.

### Görev 5: Daemon'da /api/scope

**Dosyalar:**
- Değiştir: `runtime/weightsmith_runtime/server.py` (`create_app`)
- Test: `tests/test_runtime.py`

**Arayüzler:**
- Tüketir: `make_router(resolve, list_runs)` (Görev 3)
- Üretir: `app.include_router(make_router(...), prefix="/api/scope")`. `resolve(rid)` → `store.run_dir(rid) / "scope"` varsa o dizin, yoksa `None`. `list_runs()` → `scope/` dizini olan run'lar, `(rid, dir)`, `created` sırasıyla. WS yolu `/api/scope/runs/{rid}/ws`. Vite proxy'si `/api` altında olduğu için ek ayar gerekmez.

- [ ] **Adım 1: Testler:** `test_scope_api_serves_run` (Görev 4'teki 30 adımlık run → `GET /api/scope/runs/<rid>/layers` boş olmayan bir liste), `test_scope_api_404_for_run_without_scope` (diagnostics kapalı run → 404), `test_scope_cluster_over_runs` (iki run → `GET /api/scope/cluster` iki run'ı da içerir).
- [ ] **Adım 2:** FAIL → uygula → PASS. `scripts/check.sh` → PASS. Commit: `Serve trainscope data from the daemon`.

### Görev 6: Grafikleri uPlot'a taşı

**Dosyalar:**
- Değiştir: `web/package.json` (`uplot@1.6.32`), `web/src/ui/LineChart.tsx`, `web/src/index.css` (grafik token'ları: `--series-1..7`, ızgara, bant, işaretçi; açık ve koyu)
- Oluştur: `web/src/ui/chartOptions.ts` (trainscope'un `chartOptions.js`'inin TS hâli: `toFinite`, `lossYMax`, `offScaleMarker`, `bandSpan`, `buildLineOptions`), `web/src/ui/chartOptions.test.ts` (trainscope'un `chartOptions.test.js`'inden taşınır)

**Arayüzler:**
- Üretir: `LineChart`'ın mevcut props'u (`Series[]` vb.) aynen korunur. Yeni isteğe bağlı props: `markers?: { x: number; label: string }[]`, `bands?: { x0: number; x1: number; tone: "warn" | "spike" }[]`, `syncKey?: string` (aynı anahtarı taşıyan grafiklerde imleç senkron). Tüm mevcut çağıranlar değişmeden çalışır.

- [ ] **Adım 1:** Testleri taşı ve `pnpm exec vitest run src/ui/chartOptions.test.ts` → FAIL.
- [ ] **Adım 2:** Uygula. Renkler yalnız `index.css` token'larından gelir. Tema değişince (`data-theme` / `prefers-color-scheme`) grafik yeniden çizilir.
- [ ] **Adım 3:** vitest PASS. `scripts/e2e.sh` → mevcut Metrics/Compare senaryoları PASS. Koyu temada Metrics ekran görüntüsü alınıp kontrol edilir. Commit: `Draw charts with uPlot`.

### Görev 7: web/src/scope çekirdeği

**Dosyalar:**
- Oluştur: `web/src/scope/api.ts`, `diagnosis.ts`, `layerMatrix.ts`, `spikeCluster.ts`, `runStatus.ts`, `warmup.ts`, `nodeMap.ts` ve bunların `*.test.ts`'leri. trainscope'un `__tests__/` altındaki saf mantık testleri taşınır. Bileşen testleri taşınmaz, onların yerini Görev 8-10'daki e2e senaryoları alır.
- Değiştir: `runtime/weightsmith_runtime/server.py` (`/api/runs/{rid}/files/{name}` ucunun izin listesi varsa `module_map.json` eklenir)

**Arayüzler:**
- Üretir:
  - `scopeApi(base: string, run: string)`: trainscope `api.js`'teki her `fetchX`'in TS karşılığını içeren nesne (`meta`, `global`, `layers`, `layersRanked`, `layerMatrix`, `signals`, `layer`, `moe`, `replay`, `spikes`, `spike`, `spikeLayerNames`, `spikeLayer`, `diff`). `base` Weightsmith'te `daemonBase() + "/api/scope"`, bağımsızda `"/api"`. 404 → `null` (boş durum).
  - `scopeMulti(base)`: `runs`, `compare(runs)`, `clusters(runs?)`, `counterexample(run)`.
  - `subscribeScope(base, run, onMsg)`: WS, mesaj tipleri `meta|global|global_delta|spike|layers|moe|moe_delta`. Kapatma fonksiyonu döner.
  - `nodeFor(layer: string, moduleMap: Record<string,string>): string | null`: ilk segment eşlemesi. Bilinmeyen segmentte `null`.
  - Saf fonksiyonlar trainscope'taki adlarla: `diagnoseSpike`, `groupRows`, `firstCrossings`, `sortRows`, `zColor`, `heatmapSummary`, `groupSpikes`, `spikeBands`, `runStatus`, `computeWarmupBand`.

- [ ] **Adım 1:** Testleri taşı ve `nodeMap.test.ts` yaz (`"blocks.0.weight"` → `"blocks"`'un node id'si; `"unknown.w"` → `null`). FAIL.
- [ ] **Adım 2:** Uygula, PASS, `tsc -b` temiz. Commit: `Port trainscope analysis helpers to TypeScript`.

### Görev 8: Diagnose sekmesi

**Dosyalar:**
- Oluştur: `web/src/scope/Overview.tsx`, `Layers.tsx`, `Spikes.tsx`, `Routing.tsx`, `Replay.tsx`, `WeightsDiff.tsx`, `LayerHeatmap.tsx` (canvas; trainscope'un `LayerHeatmap.jsx`'inin karşılığı), `Diagnose.tsx` (alt sekmeler, aktif run)
- Değiştir: `web/src/state/store.ts` (`BottomTab`'a `"diagnose"`), `web/src/ui/Bottom.tsx` (sekme listesi, satır 500 civarı; Metrics grafiğinde `spike` event'leri `markers` olarak)
- Test: `web/e2e/scope.mjs` (yeni senaryo, `scripts/e2e.sh` onu da koşar)

**Arayüzler:**
- Tüketir: Görev 6 (`LineChart` + `markers/bands/syncKey`), Görev 7 (tüm `scope/` modülleri)
- Üretir: `<Diagnose run={rid} onSelectLayer={(layer: string) => void} />`. Görev 9 `onSelectLayer`'ı canvas'a bağlar.

- [ ] **Adım 1: e2e senaryosunu yaz** (`scope.mjs`): örnek Transformer projesi LR rampasıyla çalıştırılır → Diagnose sekmesi açılır → Overview'da loss grafiği, heatmap canvas'ı ve en az bir sinyal satırı görünür → Spikes alt sekmesinde spike listesi dolu → diagnostics kapalı bir run seçilince "Bu run teşhis kaydı içermiyor" boş durumu görünür. FAIL.
- [ ] **Adım 2:** Görünümleri trainscope'taki karşılıklarından taşı. Yalnız Weightsmith `ui/` bileşenleri (`Select`, `Modal`, `TooltipLayer`) ve token'lar kullanılır. Routing alt sekmesi yalnız `moe` verisi varsa görünür. Canlı run'da `subscribeScope` ile artımlı güncelleme yapılır.
- [ ] **Adım 3:** e2e PASS, `scripts/check.sh` PASS. Açık ve koyu temada ekran görüntüsü kontrolü. Commit: `Add the Diagnose tab`.

### Görev 9: Canvas bağı

**Dosyalar:**
- Değiştir: `web/src/ui/Canvas.tsx`, `web/src/ui/NodeView.tsx` (boya: arka plan/çerçeve tonu, spike kaynağı işareti), `web/src/ui/Inspector.tsx` (seçili node'un katman metrikleri: grad norm ve kurtosis mini `LineChart`), `web/src/state/store.ts` (`paint: { on: boolean; metric: "grad_norm" | "kurtosis" | "weight_norm" }`, `scopeByNode: Record<string, number>`), `web/src/scope/Diagnose.tsx`
- Test: `web/e2e/scope.mjs` (genişletilir)

**Arayüzler:**
- Tüketir: `nodeFor` (Görev 7), `onSelectLayer` (Görev 8), `module_map.json`
- Üretir: Spike kaynağı node'u, spike penceresinde ilk eşiği geçen (`firstCrossings`) eğitilen katmanın node'udur.

- [ ] **Adım 1: e2e'yi genişlet:** heatmap satırına tıklayınca canvas'ta `nodeFor(layer)` node'u seçili olur. Boya anahtarı açıkken en az bir node'un `data-paint` niteliği vardır. Spike'lı run'da tam bir node `data-spike-source` taşır. LoRA projesinde `data-spike-source` dondurulmuş bir node'da olmaz. Inspector'da "Layer metrics" başlığı ve bir grafik görünür. FAIL.
- [ ] **Adım 2:** Uygula. Boya ölçeği yalnız mevcut token'lardan gelir. Eşlenemeyen katmanlar heatmap'te kalır ama tıklanınca seçim yapmaz.
- [ ] **Adım 3:** e2e PASS, `scripts/check.sh` PASS. Commit: `Link diagnostics to the canvas`.

### Görev 10: Runs ve Compare

**Dosyalar:**
- Değiştir: `web/src/ui/Bottom.tsx` (Runs tablosu: durum kolonu `runStatus` sözlüğüyle, "ilk sinyal" adımı kolonu; Compare: kümeler, ortak kader bandı, en yakın stabil karşı-örnek, config farkı ve mevcut graph diff'i yan yana)
- Test: `web/e2e/scope.mjs`

**Arayüzler:**
- Tüketir: `scopeMulti` (Görev 7), `summary.first_spike_step` (Görev 4)

- [ ] **Adım 1: e2e:** biri stabil, biri diverge eden iki run. Runs'ta durum kolonları farklı. İkisi Compare'e işaretlenince bir "neden patladı" paneli görünür ve diverge eden run'ın karşı-örneği stabil run olur. Scope'u olmayan bir run da işaretlenince panel o run'ı "kayıt yok" olarak listeler, çökmez. FAIL.
- [ ] **Adım 2:** Uygula → PASS. `scripts/check.sh` PASS. Commit: `Explain diverged runs in Runs and Compare`.

### Görev 11: Bağımsız trainscope arayüzü

**Dosyalar:**
- Oluştur: `web/scope.html`, `web/src/scope/standalone.tsx` (run listesi + `Diagnose`, canvas bağı yok, `base = "/api"`)
- Değiştir: `web/vite.config.ts` (`vite build --mode scope` → giriş `scope.html`, `outDir: ../trainscope/trainscope/ui/static`, `emptyOutDir: true`, çıktıda `index.html` adıyla), `web/package.json` (`build:scope` script'i), `trainscope/trainscope/ui/server.py` (static servis yolu gerekirse), `scripts/check.sh` (`pnpm run build:scope`, Python testlerinden önce)
- Sil: `trainscope/frontend/`
- Test: `tests/test_trainscope_boundary.py::test_standalone_install` (`uv` yoksa skip)

- [ ] **Adım 1: Test:** geçici venv'e `uv pip install ./trainscope` → `python -c "import trainscope.ui.server"` çalışır, `weightsmith_core` import edilemez, kurulu pakette `trainscope/ui/static/index.html` vardır. FAIL (static henüz build edilmedi).
- [ ] **Adım 2:** Uygula. `trainscope/pyproject.toml`'daki artifact kuralı korunur.
- [ ] **Adım 3:** PASS. Elle kontrol: temiz venv'de `trainscope demo` UI'ı açar, Runs ve Overview dolu. Commit: `Build the standalone trainscope UI from the shared web package`.

### Görev 12: CI, yayın, dokümanlar

**Dosyalar:**
- Oluştur: `.github/workflows/ci.yml` (`scripts/check.sh`, Chrome kurulumu ile), `.github/workflows/publish-trainscope.yml` (tetik: `trainscope-v*` etiketi; adımlar: `pnpm run build:scope` → `uv build trainscope/` → PyPI trusted publishing; eski `publish.yml`'den taşınır)
- Değiştir: `trainscope/pyproject.toml` (`version = "1.11.0"`, `Repository`/`Homepage` → `https://github.com/kaelvalen/weightsmith/tree/main/trainscope`), `trainscope/README.md` (rozetler, linkler, "Development" bölümü yeni konuma göre; Plotly'den söz eden cümle güncellenir), `trainscope/CHANGELOG.md` (1.11.0: "Moved into the Weightsmith repository; UI rebuilt on the shared web package; HTTP API is now keyed by run"), `README.md` (trainscope ve Diagnose bölümü), `docs/decisions.md` (bu turun özeti, Görev 4'teki overhead tablosu, bu iki dokümana link)

- [ ] **Adım 1:** Workflow'ları yaz. YAML'ı `uvx check-jsonschema --builtin-schema vendor.github-workflows` ile doğrula.
- [ ] **Adım 2:** Son kontrol: commit mesajlarında ve dosyalarda asistan adı araması → `0`. `scripts/check.sh` → PASS.
- [ ] **Adım 3:** Commit: `Release trainscope from the Weightsmith repository`.
- [ ] **Adım 4: Arda'ya sor:** Weightsmith `main`'e push; Görev 1'deki trainscope force push (henüz yapılmadıysa); eski repo README'sine yönlendirme ve arşiv; `trainscope-v1.11.0` etiketi (PyPI trusted publishing ayarı Arda'da).
