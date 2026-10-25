"""Ajoute le timecode standard OpenEXR (attributs `timeCode` et `framesPerSecond`)
aux EXR d'une séquence, à partir du numéro de frame du nom de fichier
(numérotation basée sur le timecode : frame N -> N / fps).

Les pixels ne sont pas modifiés : seuls l'en-tête et la table des offsets
(décalée de la taille ajoutée) sont réécrits. Chaque fichier est écrit dans un
fichier temporaire puis remplacé.

Usage :
    python exr_timecode.py "<dossier>" [--fps 25] [--dry-run]
"""
import argparse
import os
import re
import struct
import sys

LINES_PER_BLOCK = {0: 1, 1: 1, 2: 1, 3: 16, 4: 32, 5: 16, 6: 32, 7: 32, 8: 32, 9: 256}


def read_header(data):
    """Retourne (attributs {nom: (type, valeur)}, ordre, fin de l'en-tête)."""
    if data[:4] != b"\x76\x2f\x31\x01":
        raise ValueError("pas un fichier EXR")
    # bits 9 (tuilé), 11 (deep) et 12 (multipart) ; le bit 10 (noms longs) est accepté
    if struct.unpack("<I", data[4:8])[0] & 0x1A00:
        raise ValueError("EXR tuilé / multipart / deep non pris en charge")
    i, attrs, order = 8, {}, []
    while data[i] != 0:
        j = data.index(b"\0", i)
        name = data[i:j].decode()
        k = data.index(b"\0", j + 1)
        typ = data[j + 1:k].decode()
        size = struct.unpack("<i", data[k + 1:k + 5])[0]
        attrs[name] = (typ, data[k + 5:k + 5 + size])
        order.append(name)
        i = k + 5 + size
    return attrs, order, i + 1


def bcd(v):
    return ((v // 10) << 4) | (v % 10)


def timecode_value(frame, fps):
    ff = frame % fps
    s = frame // fps
    hh, mm, ss = (s // 3600) % 24, (s // 60) % 60, s % 60
    packed = bcd(ff) | (bcd(ss) << 8) | (bcd(mm) << 16) | (bcd(hh) << 24)
    return struct.pack("<II", packed, 0), "%02d:%02d:%02d:%02d" % (hh, mm, ss, ff)


def attr_bytes(name, typ, value):
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(value)) + value


def process(path, fps, dry_run):
    m = re.search(r"(\d+)\.exr$", path, re.I)
    if not m:
        return None
    frame = int(m.group(1))
    data = open(path, "rb").read()
    attrs, order, hdr_end = read_header(data)

    tc, tc_str = timecode_value(frame, fps)
    attrs["timeCode"] = ("timecode", tc)
    attrs["framesPerSecond"] = ("rational", struct.pack("<iI", fps, 1))
    for n in ("timeCode", "framesPerSecond"):
        if n not in order:
            order.append(n)

    x0, y0, x1, y1 = struct.unpack("<4i", attrs["dataWindow"][1])
    nblocks = -(-(y1 - y0 + 1) // LINES_PER_BLOCK[attrs["compression"][1][0]])
    offsets = struct.unpack("<%dQ" % nblocks, data[hdr_end:hdr_end + 8 * nblocks])
    if offsets[0] != hdr_end + 8 * nblocks:
        raise ValueError("table des offsets inattendue")

    header = data[:8] + b"".join(attr_bytes(n, *attrs[n]) for n in order) + b"\0"
    delta = len(header) - hdr_end
    table = struct.pack("<%dQ" % nblocks, *(o + delta for o in offsets))
    out = header + table + data[hdr_end + 8 * nblocks:]

    # Contrôle : le premier bloc pointé doit commencer à la ligne y0
    if struct.unpack("<i", out[offsets[0] + delta:offsets[0] + delta + 4])[0] != y0:
        raise ValueError("contrôle de bloc échoué")
    if not dry_run:
        tmp = path + ".tctmp"
        with open(tmp, "wb") as f:
            f.write(out)
        os.replace(tmp, path)
    return tc_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    files = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(".exr"))
    first = last = None
    for n, f in enumerate(files, 1):
        tc = process(os.path.join(args.folder, f), args.fps, args.dry_run)
        first = first or (f, tc)
        last = (f, tc)
        if n % 200 == 0:
            print(f"{n}/{len(files)}", flush=True)
    print(f"{len(files)} fichiers - premier {first} - dernier {last}"
          + (" (simulation)" if args.dry_run else ""))


if __name__ == "__main__":
    sys.exit(main())
