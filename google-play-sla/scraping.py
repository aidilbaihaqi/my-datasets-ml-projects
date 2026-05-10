import os
import time
import random

try:
    from google_play_scraper import reviews, Sort
except ImportError:
    raise ImportError(
        "\n[ERROR] Library belum terinstall.\n"
        "Jalankan: pip install google-play-scraper\n"
    )

import pandas as pd

OUTPUT_DIR  = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_comments.csv")

TARGET_PER_CLASS = 3_000   
PER_APP_GENERAL  = 3_500   
PER_APP_NEUTRAL  = 1_500   

APP_LIST = [
    {"app_id": "com.tokopedia.tkpd",   "app_name": "Tokopedia", "category": "marketplace"},
    {"app_id": "com.shopee.id",         "app_name": "Shopee",    "category": "marketplace"},
    {"app_id": "com.lazada.android",    "app_name": "Lazada",    "category": "marketplace"},
    {"app_id": "com.bukalapak.android", "app_name": "Bukalapak", "category": "marketplace"},
]

def rating_to_label(score: int) -> str:
    if score >= 4:   return "positif"
    elif score == 3: return "netral"
    else:            return "negatif"

def scrape_app(app_id: str, app_name: str, count: int = PER_APP_GENERAL) -> list:
    print(f"\n  [{app_name}] scraping semua rating ...")
    collected = []
    token     = None

    for sort_by, sort_name in [(Sort.NEWEST, "NEWEST"), (Sort.MOST_RELEVANT, "RELEVANT")]:
        try:
            result, token = reviews(
                app_id,
                lang    = "id",
                country = "id",
                sort    = sort_by,
                count   = count // 2,
            )
            valid = [r for r in result if len(str(r.get("content", "")).strip()) >= 10]
            collected.extend(valid)
            print(f"    [{sort_name}] {len(valid):,} review valid")
            time.sleep(random.uniform(1.5, 3.0))
        except Exception as e:
            print(f"    [{sort_name}] ERROR: {e}")

    while token and len(collected) < count:
        try:
            result, token = reviews(
                app_id,
                lang               = "id",
                country            = "id",
                sort               = Sort.NEWEST,
                count              = 500,
                continuation_token = token,
            )
            if not result:
                break
            valid = [r for r in result if len(str(r.get("content", "")).strip()) >= 10]
            collected.extend(valid)
            time.sleep(random.uniform(1.0, 2.0))
        except Exception as e:
            print(f"    [PAGINATION] ERROR: {e}")
            break

    print(f"    total terkumpul: {len(collected):,}")
    return collected[:count]

def scrape_app_neutral(app_id: str, app_name: str, count: int = PER_APP_NEUTRAL) -> list:
    print(f"\n  [{app_name}] scraping khusus rating 3 (netral) ...")
    collected = []
    token     = None
    attempts  = 0
    max_pages = 15 

    try:
        result, token = reviews(
            app_id,
            lang    = "id",
            country = "id",
            sort    = Sort.NEWEST,
            count   = 2000,
        )
        neutrals = [r for r in result
                    if r.get("score") == 3
                    and len(str(r.get("content", "")).strip()) >= 10]
        collected.extend(neutrals)
        time.sleep(random.uniform(1.5, 2.5))
    except Exception as e:
        print(f"    [ERROR] {e}")
        return []

    while token and len(collected) < count and attempts < max_pages:
        try:
            result, token = reviews(
                app_id,
                lang               = "id",
                country            = "id",
                sort               = Sort.NEWEST,
                count              = 500,
                continuation_token = token,
            )
            if not result:
                break
            neutrals = [r for r in result
                        if r.get("score") == 3
                        and len(str(r.get("content", "")).strip()) >= 10]
            collected.extend(neutrals)
            attempts += 1
            time.sleep(random.uniform(0.8, 1.5))
        except Exception as e:
            print(f"    [PAGINATION] ERROR: {e}")
            break

    print(f"    rating-3 terkumpul: {len(collected):,}")
    return collected[:count]

def run_scraping() -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_rows = []

    print("\n" + "=" * 60)
    print("  Play Store Scraper — E-Commerce Indonesia")
    print("  Strategi: scraping umum + booster netral (rating 3)")
    print("=" * 60)

    print("\n[FASE 1] Scraping umum semua rating ...")
    for app_info in APP_LIST:
        raw = scrape_app(app_info["app_id"], app_info["app_name"])
        for r in raw:
            text  = str(r.get("content", "")).strip()
            score = int(r.get("score", 3))
            all_rows.append({
                "app_id"    : app_info["app_id"],
                "app_name"  : app_info["app_name"],
                "category"  : app_info["category"],
                "review_id" : r.get("reviewId", ""),
                "text"      : text,
                "rating"    : score,
                "thumbs_up" : r.get("thumbsUpCount", 0),
                "at"        : str(r.get("at", "")),
                "true_label": rating_to_label(score),
            })

    df_phase1     = pd.DataFrame(all_rows).drop_duplicates(subset=["text"])
    neutral_count = (df_phase1["true_label"] == "netral").sum()
    print(f"\n  Fase 1 selesai — {len(df_phase1):,} review")
    print(f"  Netral saat ini: {neutral_count:,} (target: {TARGET_PER_CLASS:,})")

    if neutral_count < TARGET_PER_CLASS:
        needed = TARGET_PER_CLASS - neutral_count
        print(f"\n[FASE 2] Netral kurang {needed:,}, scraping booster rating-3 ...")

        existing_texts = set(df_phase1["text"].tolist())
        extra_neutral  = []

        for app_info in APP_LIST:
            if len(extra_neutral) >= needed:
                break
            per_app_need = (needed // len(APP_LIST)) + 300
            raw = scrape_app_neutral(
                app_info["app_id"],
                app_info["app_name"],
                count=per_app_need,
            )
            for r in raw:
                text = str(r.get("content", "")).strip()
                if text in existing_texts:
                    continue
                extra_neutral.append({
                    "app_id"    : app_info["app_id"],
                    "app_name"  : app_info["app_name"],
                    "category"  : app_info["category"],
                    "review_id" : r.get("reviewId", ""),
                    "text"      : text,
                    "rating"    : 3,
                    "thumbs_up" : r.get("thumbsUpCount", 0),
                    "at"        : str(r.get("at", "")),
                    "true_label": "netral",
                })
                existing_texts.add(text)

        print(f"  Booster netral: +{len(extra_neutral):,} review baru")
        all_rows.extend(extra_neutral)
    else:
        print("\n[FASE 2] Netral sudah cukup, skip booster.")

    df = (
        pd.DataFrame(all_rows)
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )

    if len(df) == 0:
        raise RuntimeError(
            "\n[ERROR] Scraping menghasilkan 0 data!\n"
            "Kemungkinan penyebab:\n"
            "  1. Tidak ada koneksi internet\n"
            "  2. IP di-block sementara — coba lagi 10 menit kemudian\n"
            "  3. Coba ganti network (WiFi ke hotspot atau sebaliknya)\n"
        )

    print(f"\n{'=' * 60}")
    print(f"  Total review unik : {len(df):,}")
    print(f"\n  Distribusi label:")
    for label, cnt in df["true_label"].value_counts().items():
        pct    = cnt / len(df) * 100
        bar    = "█" * int(pct / 2)
        status = "✅" if cnt >= TARGET_PER_CLASS else "⚠️ "
        print(f"    {status} {label:10s}: {cnt:5,}  ({pct:.1f}%)  {bar}")

    print(f"\n  Distribusi per aplikasi:")
    for app, cnt in df["app_name"].value_counts().items():
        print(f"    {app:15s}: {cnt:,}")

    min_class = df["true_label"].value_counts().min()
    if min_class < 1_000:
        print(f"\n  ⚠️  Kelas terkecil hanya {min_class:,} — pertimbangkan tambah aplikasi lain.")
    elif min_class < TARGET_PER_CLASS:
        print(f"\n  ℹ️  Kelas terkecil {min_class:,} sampel (target {TARGET_PER_CLASS:,}).")
        print("     Model sudah jauh lebih seimbang dari sebelumnya.")
    else:
        print(f"\n  ✅ Semua kelas >= {TARGET_PER_CLASS:,} — distribusi seimbang!")

    return df


def main():
    df = run_scraping()
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ Dataset disimpan: {OUTPUT_FILE}")
    print(f"     {len(df):,} review | kolom: {list(df.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()