# Weightsmith

Mimari deneyi yapan araştırmacı için görsel model programlama ortamı. Canvas, PyTorch'un
üstünde bir görünüm; kaynak gerçeği **graph IR**. Graph kur → doğrula → okunabilir PyTorch üret →
local GPU'da eğit → run'ları karşılaştır.

Plan: `Görsel ML Mimari Editörü — Proje Planı.md`. Plana göre verilen kararlar ve eklenenler:
[`docs/decisions.md`](docs/decisions.md).

## Hızlı başlangıç

```bash
scripts/dev.sh serve        # canvas'ı build eder, daemon + canvas: http://127.0.0.1:7860
scripts/dev.sh              # geliştirme: daemon :7860 + Vite :5173 (hot reload)
scripts/check.sh            # tüm testler + typecheck + build
```

Gereken: `uv`, `pnpm`, Node 22+, NVIDIA sürücüsü (CPU'da da çalışır). Deney deposu
`~/.weightsmith` altında (`WEIGHTSMITH_HOME` ile değişir).

CLI, canvas olmadan aynı yolu kullanır:

```bash
uv run weightsmith validate examples/mlp_lora.json
uv run weightsmith compile examples/mnist_mlp.job.json -o out/      # model.py, train.py, data.py, config.yaml
uv run weightsmith train examples/mnist_mlp.job.json --set optimizer.lr=3e-4
uv run weightsmith runs
uv run weightsmith diff a.json b.json
uv run weightsmith export <run id> [--merge]                         # LoRA'yı tabana katlar
```

## Mimari

```
canvas (web/)  ──yazar──▶  Graph IR (schema/)  ◀──okur──  core/ validator + shape inference
                                │                          codegen/ IR → model.py / train.py
                                └──── job spec ───────────▶ runtime/ daemon → train.py alt süreci
                                                                │ WebSocket (seq numaralı)
canvas ◀──────────── metrik / log / durum ──────────────────────┘   SQLite + dosya sistemi
```

| Dizin | İçerik |
| --- | --- |
| `schema/` | IR JSON Schema (TS tipleri buradan üretilir: `web/src/ir/schema.gen.ts`) |
| `core/` | IR yükleme, migration, kanonik form + hash, node kayıtları, validator, shape inference, diff, toplu seçim, Python node sandbox'ı |
| `codegen/` | IR → okunabilir PyTorch; `train.py` / `data.py` şablonları; meta-device smoke test |
| `runtime/` | FastAPI daemon, job kuyruğu, pause/resume/stop/checkpoint, deney deposu, ablasyon, export, CLI |
| `web/` | React Flow canvas, inspector, Cmd+K paleti, metrik/karşılaştırma/kod görünümleri |
| `examples/` | Referans graph'lar + job'lar (MNIST MLP, plan örneği MLP+LoRA, Transformer, Python node) |
| `importers/` | F5 (MVP dışı), kapsam notu |
| `tests/` | core, codegen snapshot'ları, sandbox, daemon uçtan uca, TS çapraz test fixture'ları |

**Tek doğrulayıcı kuralı.** Shape inference ve validator Python'da (`core/`). Canvas anlık geri
bildirim için TS aynasını (`web/src/ir/analyze.ts`) kullanır; ayna her değişiklikte
`scripts/gen_fixtures.py`'nin Python'dan ürettiği fixture'lara karşı test edilir
(`web/src/ir/cross.test.ts`, `tests/test_cross_ts.py`). Uyuşmazlıkta daemon'un analizi kazanır;
canvas onu tercih eder.

## Canvas kullanımı

- `Cmd/Ctrl+K`: node ekle, komutlar (LoRA'yı tüm attention q/v'ye, varyant dallandır, bloğa grupla, ablasyon, örnek projeler…)
- `Cmd/Ctrl+Enter`: Run · `Enter`: inspector'a odaklan · `Shift+Enter`: bloğu aç · `Esc`: seçimi/bloğu bırak
- `Cmd/Ctrl+Z` / `Shift+Cmd/Ctrl+Z`: IR işlem günlüğünden undo/redo
- Subgraph node'una çift tık içine girer. Hiçbir node seçili değilken inspector job config'i (veri, optimizer, precision…), sorunları ve varyant diff'ini gösterir.
- Alt şerit: Metrics (canlı eğriler, pause/resume/stop/checkpoint, ablasyon, export), Runs, Compare (eğriler + graph diff), Log, Code (üretilen `model.py`; satıra tıklamak node'u seçer).

## Durum

MVP (F0–F3) ve F4 tamam; F5 (pretrained import) ve F6'nın UI tarafı yok, protokol remote'a hazır.
Ayrıntı ve bilinen sınırlar: [`docs/decisions.md`](docs/decisions.md).
