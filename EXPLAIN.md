# What this actually is, without the jargon

If you build agents on top of LLM APIs and have never looked inside one, this page is for you.
No maths. Five minutes.

---

## What's behind the API

You use a language model like this:

```
   your agent  ──►  the API  ──►  ┌─────────┐  ──►  text
                                  │    ?    │
                                  └─────────┘
```

Nobody shows you inside the box. So I built a very small box you can see all of:

```
   ┌───────────────────────────────────────────────────────┐
   │  text ──► numbers ──► guess the next number ──► text   │
   │                            ▲                           │
   │            841,216 dials, tuned by guessing wrong      │
   │            a few hundred thousand times                │
   └───────────────────────────────────────────────────────┘
```

That's the whole thing. Every frontier model you call by API is this, scaled up.
Same three steps. More dials.

---

## The five steps

```
 ①  WRITE THE TEXTBOOK
     I generate 100,000 SQL queries myself, from rules I wrote.

         SELECT region , SUM ( qty ) FROM sales GROUP BY region ;

     ↳ Because I wrote the rules, I know exactly what's in the training data.
       No scraping, no licensing, no mystery.


 ②  TURN WORDS INTO NUMBERS                        ← "tokenization"

         SELECT  region   ,   SUM  ...
           130     147    99   131

     ↳ The model never sees letters. It sees a list of integers.
       This is why LLMs can't count the r's in "strawberry" — they never
       received the letters, only a few token numbers.


 ③  PLAY "GUESS THE NEXT WORD", 3,000 ROUNDS       ← "training"

         SELECT region , SUM ( qty ) FROM ?
                                            ↳ model guesses "customers"
                                            ↳ real answer was "sales"
                                            ↳ nudge all 841,216 dials slightly

     ↳ Repeat. That is 100% of what training is. There is no other magic step.


 ④  LET IT WRITE ON ITS OWN                        ← "inference"

         SELECT plan , AVG ( spend ) FROM customers GROUP BY plan ;

     ↳ It made that up. Not copied from the textbook.


 ⑤  CHECK IT — BY ACTUALLY RUNNING THE SQL

     Run every generated query against a real database. It works, or it errors.

     ↳ No opinions. No "looks good to me." This is the step nearly every
       tutorial skips, and it's the only reason any number here means anything.
```

---

## The one clever thing hidden in the data

Every one of the 100,000 training queries obeys a rule I planted:

```
   SELECT region , SUM ( qty ) FROM sales GROUP BY region
          ▲                                         │
          └────────── must be the same word ────────┘
                        (8 words apart)
```

Why that matters: to get this right, the model has to **look back** at something it wrote eight
words earlier. Looking back is the entire reason the "attention" mechanism exists.

I planted the rule so that later I could go and check whether the model really learned to look
back — and find exactly *where* inside it that happens.

---

## The three results

```
 ①  IT WORKS

     100 out of 100 queries it writes actually run against a real database.
     A dumb "what word usually follows this word" baseline: 4 out of 100.

     ↳ A model smaller than a phone photo writes working SQL.


 ②  YOU NEED LESS MODEL THAN YOU THINK — UNTIL SUDDENLY YOU DON'T

       25,000 dials  →  writes SQL that runs        ✓
                     →  gets the look-back rule?    ✗   (random guessing)

      125,000 dials  →  writes SQL that runs        ✓
                     →  gets the look-back rule?    ✓   (every single time)

    5,000,000 dials  →  no better than 125,000. At all.

     ↳ Grammar is cheap. Meaning costs about 5x more. After that you are
       paying for nothing.
     ↳ This is the "fine-tune something small vs. just call GPT-5" decision,
       turned into a measurement instead of an opinion.


 ③  I FOUND THE ACTUAL WIRE THAT DOES THE LOOKING BACK

     The model has 16 "attention heads". I checked all 16.
     Exactly one of them stares straight back at the SELECT column.

     ↳ One specific part of the model learned one specific rule.
       Nobody programmed it. It fell out of guessing the next word.
```

---

## The most useful result: watching it hallucinate

The model gets the look-back rule right **100% of the time** on column names it saw in training.

So I hid three column names from that position during training — the words exist elsewhere in the
data, they just never appear right after `GROUP BY`. Then I asked for them:

```
   asked for:   ... GROUP BY  →  "channel"     (never seen in this position)
   it answered: ... GROUP BY  →  "carrier"     (familiar, confident, wrong)
```

Every time. Confidently.

**That is what hallucination is.** Not a bug, not a glitch — the model reaching for the familiar
thing when the correct thing is unfamiliar. Here it happens in a model small enough to point at the
exact cause.

If your agents hallucinate on you, this is the mechanism. Same one. Just bigger.

---

## Why the numbers are trustworthy

Anyone can claim a model works. These are the things that make it checkable:

- **A baseline.** 100% means nothing until you know a dumb model gets 4%.
- **Executed, not eyeballed.** Every query is run against SQLite.
- **A seed.** Run it yourself, get the same numbers to the decimal.
- **A held-out test.** Three cases deliberately hidden from training.
- **A published negative.** The model fails that test, and I say so.
- **A control experiment.** I had a theory for *why* the 125,000-dial model succeeded — that it
  needed two layers to do the look-back. I tested it with a one-layer model of the same size.
  My theory was wrong. It's in the repo.

---

## Try it

```bash
python inference.py                  # write some SQL
python tiny_gpt.py --explain         # see the numbers, the mask, the probabilities
python evaluate.py --eval            # run the generated SQL against a database
```

Full detail, all numbers, and the code: [README.md](README.md)
