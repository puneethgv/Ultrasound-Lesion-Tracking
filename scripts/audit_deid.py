"""Audit burned-in-PHI removability: confirm every DICOM declares an ultrasound region so the
region-based de-identification never silently falls back (which could leave banners in place).

Reads headers only (fast). Prints coverage and the distribution of region rectangles.
"""

from __future__ import annotations

import warnings
from collections import Counter

import pydicom

from usloc import config as C
from usloc.data.cohort import build_cohort
from usloc.data.discover import find_case_dir, list_dicoms
from usloc.deid import ultrasound_region

warnings.simplefilter("ignore")


def main() -> None:
    cohort = build_cohort()
    total = with_region = 0
    missing: list[str] = []
    rects: Counter = Counter()
    for case in cohort["case"]:
        cd = find_case_dir(case)
        if cd is None:
            continue
        for dcm in list_dicoms(cd):
            total += 1
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
            reg = ultrasound_region(ds)
            if reg is None:
                missing.append(f"{case}/{dcm.name}")
            else:
                with_region += 1
                rects[(reg.x0, reg.y0, reg.x1, reg.y1)] += 1
    print(f"DICOM files scanned : {total}")
    print(f"with US region      : {with_region} ({100*with_region/max(total,1):.1f}%)")
    print(f"WITHOUT region      : {len(missing)}  <- would use geometric fallback")
    for m in missing[:20]:
        print("   ", m)
    print("\nTop region rectangles (x0,y0,x1,y1) -> count:")
    for rect, n in rects.most_common(10):
        print(f"   {rect} -> {n}")


if __name__ == "__main__":
    main()
