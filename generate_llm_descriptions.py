import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# IMPROVED PROMPT — v3
# Key changes over v2:
#   - Generates TWO separate descriptions per AU:
#       "healthy"    → what DDCA/CDCA sees in a healthy face
#       "parkinson"  → what DDCA/CDCA sees in a PD face
#     Previously both were blended into one embedding, which
#     averaged out the discriminative signal. Now each modality
#     gets its own clean embedding for independent cross-attention.
#   - Each description is self-contained — no cross-references.
#   - Still quantitative: amplitude %, speed ms, frequency Hz.
#   - response_format=json_object enforces clean JSON output.
#   - temperature=0.2 for maximum clinical precision.
# ============================================================

COMMON_PROMPT = """You are a clinical movement-disorder specialist and expert in the Facial Action Coding System (FACS).

Your task is to write TWO separate observational descriptions for a single Facial Action Unit (AU).
These descriptions are used as semantic embeddings in a vision-language deep learning model that classifies faces as Healthy or Parkinson's disease (PD).

OUTPUT FORMAT — return valid JSON exactly like this, with no extra text:
{
  "healthy": "<description of this AU activating in a healthy adult face>",
  "parkinson": "<description of this AU as it appears in a Parkinson's disease face with hypomimia>"
}

=== RULES FOR "healthy" ===
1. Describe exactly what a trained observer sees on the face surface when this AU activates normally.
2. Use precise anatomical terms: glabella, nasolabial fold, palpebral aperture, philtrum, vermillion border, mentolabial sulcus, lip commissure, orbital rim, zygomatic arch, etc.
3. Include QUANTITATIVE movement quality: peak amplitude in mm, onset speed in ms, bilateral symmetry, and return-to-rest smoothness.
4. 3 to 4 sentences. Fluent observational prose. No bullet points.

=== RULES FOR "parkinson" ===
1. Describe exactly what a trained observer sees on the face surface for this SAME AU in a Parkinson's patient.
2. You MUST include at least ONE specific quantitative contrast from this list:
   - Amplitude reduction: "reduced by 40–60% from healthy baseline"
   - Speed reduction: "onset delayed 150–300 ms beyond normal"
   - Frequency reduction: "spontaneous rate drops from ~15/min to ~3/min"
   - Asymmetry: "left-right asymmetry exceeds 30% vs <10% in healthy"
   - Incomplete excursion: "plateau at only 50% of full movement range"
   - Rigidity: "resting muscle tone chronically elevated, compressing dynamic range"
3. End with one sentence that states the single most visually detectable difference — this sentence is the primary alignment signal for cross-attention.
4. 3 to 4 sentences. Fluent observational prose. No bullet points."""

AU_SEEDS = {
    "AU01": "AU01 — Inner Brow Raiser. Muscle: Frontalis (medial portion). Region: Medial brow, inner forehead, glabella. The inner brow corners rise obliquely, creating medial forehead furrows above the glabella.",
    "AU02": "AU02 — Outer Brow Raiser. Muscle: Frontalis (lateral portion). Region: Lateral brow, outer forehead. The outer brow tail lifts upward and outward, producing lateral forehead wrinkles.",
    "AU04": "AU04 — Brow Lowerer. Muscle: Corrugator supercilii, depressor supercilii. Region: Glabella, medial brow. The brows are pulled downward and medially, producing vertical glabellar furrows. This is an early hypomimia marker.",
    "AU05": "AU05 — Upper Lid Raiser. Muscle: Levator palpebrae superioris. Region: Upper eyelid, palpebral aperture. The upper eyelid elevates, widening the palpebral aperture and exposing more sclera.",
    "AU06": "AU06 — Cheek Raiser. Muscle: Orbicularis oculi (orbital). Region: Cheek mass, lower eyelid, outer canthus. The cheek mass rises, narrowing the lower eyelid and producing crow's feet. Combined with AU12 forms the Duchenne smile.",
    "AU07": "AU07 — Lid Tightener. Muscle: Orbicularis oculi (palpebral). Region: Lower eyelid margin. The lower eyelid tightens upward, slightly narrowing the palpebral aperture from below.",
    "AU45": "AU45 — Blink. Muscle: Orbicularis oculi (full closure). Region: Both eyelids. Complete bilateral eyelid closure and reopening. Healthy blink rate is 12–20/min. PD reduces this dramatically — the single most sensitive early biomarker.",
    "AU09": "AU09 — Nose Wrinkler. Muscle: Levator labii superioris alaeque nasi. Region: Nasal wings, nasal bridge, upper lip. Nasal wings flare laterally and upward, producing bunny lines across the nasal bridge.",
    "AU10": "AU10 — Upper Lip Raiser. Muscle: Levator labii superioris. Region: Upper lip, mid-face, nasolabial fold. The central upper lip elevates, deepening the nasolabial fold and partially exposing upper teeth.",
    "AU12": "AU12 — Lip Corner Puller. Muscle: Zygomaticus major. Region: Lip corners, nasolabial fold, cheek. The lip corners pull upward and laterally forming a smile, deepening nasolabial folds. Severely reduced in PD hypomimia.",
    "AU14": "AU14 — Dimpler. Muscle: Buccinator. Region: Lip corners, cheek. The cheeks compress inward, pulling lip corners laterally and creating cheek dimples without the upward diagonal pull of AU12.",
    "AU15": "AU15 — Lip Corner Depressor. Muscle: Depressor anguli oris. Region: Lip corners, chin. The lip corners are pulled downward, producing a sad mouth curve with skin below the corners drawn down.",
    "AU20": "AU20 — Lip Stretcher. Muscle: Risorius, platysma. Region: Lip corners, cheek, neck. The lip corners stretch laterally in a horizontal vector, distinguishable from AU12 by the absence of upward diagonal movement.",
    "AU23": "AU23 — Lip Tightener. Muscle: Orbicularis oris. Region: Lip margin. The lip margin constricts and firms, producing a narrowed and slightly thinned lip outline. Chronically elevated in PD due to perioral rigidity.",
    "AU24": "AU24 — Lip Pressor. Muscle: Orbicularis oris. Region: Lip margin. The lips press firmly together, thinning the lip line and producing a slight forward protrusion. May be chronically elevated in PD.",
    "AU25": "AU25 — Lips Part. Muscle: Depressor labii or orbicularis relaxation. Region: Lip margin, lip gap. The lips separate as the lower lip depresses, opening the lip gap. Present in speech and eating.",
    "AU26": "AU26 — Jaw Drop. Muscle: Masseter relaxation, digastric. Region: Jaw, lower face. The mandible lowers, increasing vertical lower face dimension. PD masticatory rigidity reduces jaw opening range.",
    "AU17": "AU17 — Chin Raiser. Muscle: Mentalis. Region: Chin boss, mental crease. The chin boss pushes upward, bunching chin skin and deepening the mentolabial sulcus, producing an orange-peel chin texture."
}


def generate_descriptions():
    descriptions = {}
    print(f"Starting LLM description generation (v3) for {len(AU_SEEDS)} AUs...")
    print("Key improvement: SEPARATE healthy / parkinson descriptions per AU\n")

    for au, seed in AU_SEEDS.items():
        print(f"  Generating for {au}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": COMMON_PROMPT},
                    {"role": "user",   "content": seed}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}   # enforce clean JSON
            )
            raw    = response.choices[0].message.content
            parsed = json.loads(raw)

            if "healthy" not in parsed or "parkinson" not in parsed:
                raise ValueError(f"Missing keys: {list(parsed.keys())}")

            descriptions[au] = {
                "healthy":   parsed["healthy"],
                "parkinson": parsed["parkinson"]
            }
            print(f"    ✅ {au} — healthy: {len(parsed['healthy'].split())}w  "
                  f"| parkinson: {len(parsed['parkinson'].split())}w")

        except Exception as e:
            print(f"    ❌ Error for {au}: {e}")
            descriptions[au] = {
                "healthy":   f"Error: {e}",
                "parkinson": f"Error: {e}"
            }

    with open("au_descriptions.json", "w") as f:
        json.dump(descriptions, f, indent=4)

    print(f"\nDone! Saved {len(descriptions)} AU pairs to au_descriptions.json")
    print("Each AU now has separate 'healthy' and 'parkinson' description fields.")
    print("Next: run contextualize_text_embeddings.py")


if __name__ == "__main__":
    generate_descriptions()
