/**
 * Arabic lip-sync module for TalkingHead (https://github.com/met4citizen/TalkingHead).
 *
 * Converts an Arabic word directly into Oculus visemes, following "Appendix C: Create a new
 * lip-sync module" of the TalkingHead README (grapheme -> phoneme -> viseme, the approach used for
 * Finnish). Arabic spelling is almost phonemic when the word carries its diacritics (tashkeel),
 * so every letter and vowel mark is mapped to the mouth shape of the sound it represents:
 *
 *   - consonants by place of articulation (lips, teeth, tongue tip, velum, ...),
 *   - short vowels from the diacritics  (fatha -> aa, kasra -> I, damma -> U),
 *   - long vowels from ا / ى / و / ي, diphthongs (ay, aw), tanween (+n), shadda (longer consonant),
 *   - the definite article ال, including the sun letters (الشمس -> ash-shams, no "l"),
 *   - hamza seats (أ إ آ ؤ ئ), taa marbuta (ة), alif maqsura (ى), silent alif of plural verbs (كتبوا).
 *
 * Text without diacritics (the usual case) is still lip-synced: the missing short vowels are
 * restored with a syllable heuristic (Arabic syllables are CV, CVV or CVC), so the mouth opens
 * between consonants instead of jumping from one consonant to the next.
 *
 * Usage: new TalkingHead(el, { lipsyncModules: ["ar"], lipsyncLang: "ar" }). The file must be in the
 * same folder as talkinghead.mjs.
 */

// ---------------------------------------------------------------- characters
const FATHA = "َ", DAMMA = "ُ", KASRA = "ِ", SUKUN = "ْ", SHADDA = "ّ";
const FATHATAN = "ً", DAMMATAN = "ٌ", KASRATAN = "ٍ";
const DAGGER_ALIF = "ٰ", MADDA = "ٓ", HAMZA_ABOVE = "ٔ", HAMZA_BELOW = "ٕ";
const MARKS = /[ً-ٰٟ]/;

// Consonant -> viseme (Oculus set: sil PP FF TH DD kk CH SS nn RR aa E I O U)
const CONSONANT = {
  "ب": "PP", "م": "PP",                          // lips closed
  "ف": "FF",                                     // lower lip to upper teeth
  "ث": "TH", "ذ": "TH", "ظ": "TH",               // tongue between the teeth
  "ت": "DD", "د": "DD", "ط": "DD", "ض": "DD",    // tongue tip on the ridge
  "ن": "nn", "ل": "nn",                          // n, l
  "ر": "RR",
  "س": "SS", "ص": "SS", "ز": "SS",
  "ش": "CH", "ج": "CH",
  "ك": "kk", "ق": "kk", "غ": "kk", "خ": "kk",    // back of the tongue: jaw slightly open
  "ح": "GUTTURAL", "ع": "GUTTURAL",              // throat sounds: the jaw stays open
  "ه": "GLOTTAL", "ء": "GLOTTAL",                // h, glottal stop: no mouth shape of their own
};
const SUN_LETTERS = new Set([..."تثدذرزسشصضطظلن"]);
const PREFIXES = ["وبال", "وكال", "فبال", "فال", "وال", "بال", "كال", "لل", "ال"];

// Relative durations (1 = an average sound); TalkingHead scales them to the real word duration
const DUR = {
  PP: 0.9, FF: 1.0, TH: 1.0, DD: 0.8, kk: 0.8, CH: 1.0, SS: 1.1, nn: 0.8, RR: 0.8,
  GUTTURAL: 0.6, GLOTTAL: 0.4, GLIDE: 0.6,
  SHORT: 0.9, LONG: 1.6, GUESS: 0.7,   // GUESS = short vowel restored by the heuristic
};

// ---------------------------------------------------------------- numbers
const ONES = ["", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"];
const TEENS = ["عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر", "خمسة عشر", "ستة عشر",
  "سبعة عشر", "ثمانية عشر", "تسعة عشر"];
const TENS = ["", "عشرة", "عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون"];
const HUNDREDS = ["", "مائة", "مائتان", "ثلاثمائة", "أربعمائة", "خمسمائة", "ستمائة", "سبعمائة",
  "ثمانمائة", "تسعمائة"];

function below1000(n) {
  const parts = [];
  if (n >= 100) parts.push(HUNDREDS[Math.floor(n / 100)]);
  const r = n % 100;
  if (r >= 20) parts.push(r % 10 ? `${ONES[r % 10]} و${TENS[Math.floor(r / 10)]}` : TENS[r / 10]);
  else if (r >= 10) parts.push(TEENS[r - 10]);
  else if (r > 0) parts.push(ONES[r]);
  return parts.join(" و");
}

function numberToWords(num) {
  let n = parseInt(num, 10);
  if (!Number.isFinite(n)) return num;
  if (n === 0) return "صفر";
  if (n >= 1e9) return [...String(num)].map((d) => ONES[+d] || "صفر").join(" ");
  const parts = [];
  const millions = Math.floor(n / 1e6); n %= 1e6;
  const thousands = Math.floor(n / 1000); n %= 1000;
  if (millions) parts.push(millions === 1 ? "مليون" : millions === 2 ? "مليونان" : `${below1000(millions)} ${millions <= 10 ? "ملايين" : "مليون"}`);
  if (thousands) parts.push(thousands === 1 ? "ألف" : thousands === 2 ? "ألفان" : `${below1000(thousands)} ${thousands <= 10 ? "آلاف" : "ألف"}`);
  if (n) parts.push(below1000(n));
  return parts.join(" و");
}

// ---------------------------------------------------------------- the module
export class LipsyncAr {
  constructor() {
    this.symbols = { "%": " بالمائة ", "&": " و ", "+": " زائد ", "$": " دولار ", "€": " يورو ", "=": " يساوي " };
  }

  /**
   * Prepare text for both speech and lip sync: digits and symbols to words, unspoken characters
   * removed. Diacritics are kept because they tell which vowel is spoken.
   * @param {string} s Text
   * @return {string} Pre-processed text
   */
  preProcessText(s) {
    if (!s) return "";
    return s
      .normalize("NFC")
      .replace(/ـ/g, "")                                          // tatweel
      .replace(/[٠-٩]/g, (d) => String(d.charCodeAt(0) - 0x660))  // Arabic-Indic digits
      .replace(/[۰-۹]/g, (d) => String(d.charCodeAt(0) - 0x6f0))
      .replace(/[%&+$€=]/g, (c) => this.symbols[c])
      .replace(/(\d)[.,٫](\d)/g, "$1 فاصلة $2")
      .replace(/\d+/g, (m) => ` ${numberToWords(m)} `)
      .replace(/[^\p{L}\p{M}\s'-]/gu, " ")                        // punctuation, emojis, ...
      .replace(/\s+/g, " ")
      .trim();
  }

  /**
   * Convert one word into Oculus visemes with relative start times and durations.
   * @param {string} w Word (already pre-processed)
   * @return {{words:string, visemes:string[], times:number[], durations:number[]}}
   */
  wordsToVisemes(w) {
    const out = { words: w, visemes: [], times: [], durations: [] };
    let t = 0;
    const emit = (viseme, d) => {
      const last = out.visemes.length - 1;
      if (last >= 0 && out.visemes[last] === viseme) {
        out.durations[last] += d;           // same mouth shape: hold it longer
      } else {
        out.visemes.push(viseme);
        out.times.push(t);
        out.durations.push(d);
      }
      t += d;
    };
    const pause = (d) => { t += d; };

    for (const word of w.split(/[\s'-]+/)) {
      if (!word) continue;
      for (const s of this.phonemes(word)) {
        if (s.v === "GLOTTAL") pause(DUR.GLOTTAL);
        else if (s.v === "GUTTURAL") emit("aa", DUR.GUTTURAL * s.k);   // open jaw for ح ع
        else emit(s.v, s.d * s.k);
      }
    }
    return out;
  }

  /**
   * Arabic word -> list of sounds {v: viseme, d: duration, k: length factor}.
   * @param {string} word
   * @return {{v:string, d:number, k:number}[]}
   */
  phonemes(word) {
    const units = this.units(word);
    if (!units.length) return [];
    this._lastUnit = units[units.length - 1];
    const vocalised = units.some((u) => u.marks.length);
    const sounds = [];
    const vowel = (v, d) => sounds.push({ v, d, k: 1 });
    const consonant = (ch, k = 1) => {
      const v = CONSONANT[ch];
      if (v) sounds.push({ v, d: DUR[v] ?? 0.8, k });
    };

    // 1. definite article and the prefixes that come before it (و ف ب ك ل)
    let i = 0;
    const plain = units.map((u) => u.ch).join("");
    const prefix = PREFIXES.find((p) => plain.startsWith(p) && plain.length > p.length + 1);
    if (prefix) {
      const lead = prefix.slice(0, prefix.indexOf("ال") >= 0 ? prefix.indexOf("ال") : prefix.length - 1);
      for (const ch of lead) {                       // wa-, fa-, bi-, ka-, li-
        if (ch === "و") sounds.push({ v: "U", d: DUR.GLIDE, k: 1 });
        else consonant(ch);
        vowel(ch === "ب" || ch === "ل" ? "I" : "aa", DUR.SHORT);
      }
      i = prefix.length;
      const next = units[i]?.ch;
      if (prefix === "لل") {
        // li + al: the lam of the article is pronounced unless a sun letter follows
        if (!SUN_LETTERS.has(next)) consonant("ل");
      } else if (!lead.length) {
        vowel("aa", DUR.SHORT);                      // a- of "al" at the start of speech
        if (!SUN_LETTERS.has(next)) consonant("ل");
      } else if (!SUN_LETTERS.has(next)) {
        consonant("ل");                              // wa-l-kitab: the alif is silent
      }
      if (SUN_LETTERS.has(next)) units[i].geminate = true;   // ash-shams: doubled sun letter
    }

    // 2. letters
    // Positions of the vowels restored for undiacritised text (see restoreVowels)
    const guessed = vocalised ? new Set() : this.restoreVowels(units, i);
    for (; i < units.length; i++) {
      const u = units[i];
      const prev = units[i - 1];
      const next = units[i + 1];
      const last = i === units.length - 1;
      const k = u.marks.includes(SHADDA) || u.geminate ? 1.8 : 1;

      switch (u.ch) {
        case "ا":
          // long a, except the silent alif of plural verbs (كتبوا) and the alif after tanween fatha
          if (last && prev?.ch === "و" && i >= 3) break;
          if (prev?.marks.includes(FATHATAN) || u.marks.includes(FATHATAN)) { vowel("aa", DUR.SHORT); consonant("ن"); break; }
          if (i === 0) { vowel("aa", DUR.SHORT); break; }  // hamzat wasl at the start
          vowel("aa", DUR.LONG);
          break;
        case "ى":
          vowel("aa", DUR.LONG);
          break;
        case "آ":
          consonant("ء"); vowel("aa", DUR.LONG);
          break;
        case "أ": case "إ": case "ؤ": case "ئ":
          consonant("ء");
          this.vowelMarks(u, vowel, consonant, u.ch === "إ" ? "I" : u.ch === "ؤ" ? "U" : u.ch === "ئ" ? "I" : "aa", i === 0 || guessed.has(i));
          break;
        case "ة":
          // taa marbuta: "-a" at the end of a phrase, "-at" + vowel when it carries a vowel mark
          if (u.marks.some((m) => m !== SUKUN)) { consonant("ت"); this.vowelMarks(u, vowel, consonant, null, false); }
          else if (!prev?.marks.includes(FATHA)) vowel("aa", DUR.SHORT);
          break;
        case "و":
        case "ي": {
          const glide = u.ch === "و" ? "U" : "I";
          const isLong = this.isLongVowel(u, prev, next, i, vocalised);
          if (isLong === "long") {
            vowel(glide, DUR.LONG);
          } else if (isLong === "diphthong") {
            vowel(glide, DUR.GLIDE);                 // the a- is already there: ay, aw
          } else {
            sounds.push({ v: glide, d: DUR.GLIDE, k });   // consonant w / y
            this.vowelMarks(u, vowel, consonant, null, guessed.has(i));
          }
          break;
        }
        default:
          if (!CONSONANT[u.ch]) break;               // not an Arabic letter
          consonant(u.ch, k);
          this.vowelMarks(u, vowel, consonant, null, guessed.has(i));
      }
    }
    return sounds;
  }

  /** Short vowel and tanween written on a letter, or a restored vowel when `guess` is set. */
  vowelMarks(u, vowel, consonant, fallback, guess) {
    const m = u.marks;
    if (m.includes(FATHA) || m.includes(DAGGER_ALIF)) vowel("aa", m.includes(DAGGER_ALIF) ? DUR.LONG : DUR.SHORT);
    else if (m.includes(KASRA)) vowel("I", DUR.SHORT);
    else if (m.includes(DAMMA)) vowel("U", DUR.SHORT);
    else if (m.includes(FATHATAN)) { vowel("aa", DUR.SHORT); consonant("ن"); }
    else if (m.includes(KASRATAN)) { vowel("I", DUR.SHORT); consonant("ن"); }
    else if (m.includes(DAMMATAN)) { vowel("U", DUR.SHORT); consonant("ن"); }
    else if (m.includes(SUKUN)) { /* no vowel */ }
    else if (guess) vowel(fallback || "aa", DUR.GUESS);
    else if (fallback) vowel(fallback, DUR.SHORT);
  }

  /** Is و / ي a long vowel (uu, ii), the second half of a diphthong (aw, ay) or a consonant (w, y)? */
  isLongVowel(u, prev, next, i, vocalised) {
    if (i === 0) return "consonant";                               // word-initial w / y
    if (u.ch === "و" && next?.ch === "ا" && !next.marks.length && i >= 2 && next === this._lastUnit) {
      return "long";                                               // كتبوا: -uu + silent alif
    }
    const nextIsVowelLetter = next && "اىة".includes(next.ch);
    if (vocalised) {
      if (u.marks.some((m) => [FATHA, KASRA, DAMMA, SHADDA, FATHATAN, KASRATAN, DAMMATAN].includes(m))) return "consonant";
      if (prev?.marks.includes(FATHA)) return "diphthong";
      return "long";
    }
    if (nextIsVowelLetter) return "consonant";                     // w / y followed by a vowel
    if (!prev || !CONSONANT[prev.ch]) return "consonant";
    return "long";
  }

  /**
   * Undiacritised text: decide after which consonants a short vowel is spoken.
   * Arabic syllables are CV, CVV or CVC, so inside a run of consonants every other consonant
   * carries a vowel; the last consonant of a run takes the following long vowel, and a run at
   * the end of the word keeps the final consonant vowel-less (CaCaC, maCCaC).
   * @return {Set<number>} unit indexes that get a restored vowel
   */
  restoreVowels(units, start) {
    const guessed = new Set();
    const vowelLetter = (u, idx) => {
      if (!u) return false;
      if ("اىةآ".includes(u.ch)) return true;
      if (u.ch === "و" && idx >= 2 && idx === units.length - 2 && units[idx + 1].ch === "ا") return true;  // كتبوا
      if ((u.ch === "و" || u.ch === "ي") && idx > start && CONSONANT[units[idx - 1]?.ch] &&
          !(units[idx + 1] && "اىة".includes(units[idx + 1].ch))) return true;
      return false;
    };
    let run = [];
    const flush = (atEnd) => {
      const n = run.length;
      if (!n) return;
      if (atEnd) {
        if (n === 2) guessed.add(run[0]);                          // CaC
        else if (n === 3) { guessed.add(run[0]); guessed.add(run[1]); }  // CaCaC
        else for (let j = 0; j < n - 1; j += 2) guessed.add(run[j]);     // maC.CaC
      } else if (n === 3) {
        guessed.add(run[0]); guessed.add(run[1]);                  // CaCaC+long vowel (katabuu)
      } else {
        for (let j = n - 2; j >= 0; j -= 2) guessed.add(run[j]);   // ...Ca C+long vowel
      }
      run = [];
    };
    for (let idx = start; idx < units.length; idx++) {
      const u = units[idx];
      if (vowelLetter(u, idx)) flush(false);
      else if (CONSONANT[u.ch] || "أإؤئوي".includes(u.ch)) run.push(idx);   // و ي here are w, y
    }
    flush(true);
    return guessed;
  }

  /** Split a word into letters with the diacritics written on each of them. */
  units(word) {
    const units = [];
    for (const ch of word.normalize("NFC")) {
      if (MARKS.test(ch) || ch === MADDA || ch === HAMZA_ABOVE || ch === HAMZA_BELOW) {
        if (units.length) units[units.length - 1].marks.push(ch);
      } else if (/\p{L}/u.test(ch)) {
        units.push({ ch, marks: [] });
      }
    }
    return units;
  }

  /** Viseme names this module can produce. */
  getVisemeNames() {
    return ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U"];
  }
}
