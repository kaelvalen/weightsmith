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

## Bilinçli olarak yapılmayanlar / sonraki işler

- **F5 pretrained import** ve opak blok node'u: yok (`importers/README.md`).
- **F6 remote**: protokol hazır (daemon tek API, veri URI ile, WS devam), UI'da daemon adresi seçimi yok. Şimdilik SSH tüneli + `WEIGHTSMITH_DAEMON=http://... pnpm dev`.
- **Custom adapter** (`kind: custom`): IR, validator ve codegen destekliyor; inspector'da düzenleme UI'ı yok (JSON'dan yazılabilir).
- İç içe subgraph, çoklu pencere senkronu, Tauri paketi, `torch.compile` dışında kernel node'ları yok.
- Otomatik yerleşim katmanlı (longest-path); residual edge'ler zincirle aynı hizaya düşebiliyor, elle taşımak gerekiyor.
- Monaco ~4 MB; yalnızca kod görünümü ve Python node editörü açılınca yüklenir.
