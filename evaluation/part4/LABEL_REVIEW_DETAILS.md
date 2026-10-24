# Part 4 relevance-label review: row-by-row decisions

All 502 query/track pairs were explicitly reviewed by Codex using catalog metadata. These are AI judgments requiring human confirmation; no listening or independent blind assessment occurred.

Full tags, context and original grades are preserved in `results/reviewed_judgments.csv`. Track keys map to UUIDs in `results/label_review_audit.json`. Grades are chosen in `label_review_decisions.json`; the export script only validates and materializes those choices.

## tobi-g01: Recommend alternative rock music

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T078: Double Pump | Girl Talk | 2 → 0 | Electronic/hip-hop mashup, indie pop, or electronic metadata does not substantiate the requested alternative-rock style. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T027: Isolation | Alter Bridge | 2 → 2 | Alternative-rock evidence supports the requested style; coherent alternative plus rock/indie-rock tags also qualify. |
| T157: Any Advice? | Nonpoint | 2 → 2 | Alternative-rock evidence supports the requested style; coherent alternative plus rock/indie-rock tags also qualify. |
| T183: Revolution Rock | The Clash | 2 → 1 | Related rock is supported, but reggae/classic/progressive/Latin/folk/hip-hop crossover is a weaker fit to the specific alternative-rock request. |
| T080: Chloe | GROUPLOVE, Grouplove | 2 → 2 | Alternative-rock evidence supports the requested style; coherent alternative plus rock/indie-rock tags also qualify. |
| T171: Price To Play | Staind | 2 → 2 | Alternative-rock evidence supports the requested style; coherent alternative plus rock/indie-rock tags also qualify. |
| T056: Uh Oh Hello | Elefant | 2 → 2 | Alternative-rock evidence supports the requested style; coherent alternative plus rock/indie-rock tags also qualify. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 2 → 0 | Electronic/hip-hop mashup, indie pop, or electronic metadata does not substantiate the requested alternative-rock style. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 2 → 1 | Related rock is supported, but reggae/classic/progressive/Latin/folk/hip-hop crossover is a weaker fit to the specific alternative-rock request. |
| T158: Television, Television | OK Go | 2 → 2 | Alternative-rock evidence supports the requested style; coherent alternative plus rock/indie-rock tags also qualify. |
| T134: Ángel de amor | Maná | 2 → 1 | Related rock is supported, but reggae/classic/progressive/Latin/folk/hip-hop crossover is a weaker fit to the specific alternative-rock request. |
| T068: Suggestion | Fugazi | 2 → 2 | Alternative-rock evidence supports the requested style; coherent alternative plus rock/indie-rock tags also qualify. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | Electronic/hip-hop mashup, indie pop, or electronic metadata does not substantiate the requested alternative-rock style. |
| T185: Take It As It Comes | The Doors | 2 → 1 | Related rock is supported, but reggae/classic/progressive/Latin/folk/hip-hop crossover is a weaker fit to the specific alternative-rock request. Review caveat: Classic/psychedelic rock dominates despite one alternative-rock tag; partial alternative-rock or Linkin Park similarity is more defensible than full credit. |
| T104: Failure | Laura Marling | 2 → 1 | Related rock is supported, but reggae/classic/progressive/Latin/folk/hip-hop crossover is a weaker fit to the specific alternative-rock request. Review caveat: Folk/acoustic/singer-songwriter evidence dominates despite folk-rock/pop/rock tags; contextual relevance is partial. |
| T154: Rock N Roll | Mos Def | 2 → 1 | Related rock is supported, but reggae/classic/progressive/Latin/folk/hip-hop crossover is a weaker fit to the specific alternative-rock request. |
| T139: Shame | Matchbox Twenty, Archilla, Matt | 2 → 2 | Alternative-rock evidence supports the requested style; coherent alternative plus rock/indie-rock tags also qualify. |
| T163: Get Right | Pearl Jam | 2 → 2 | Alternative-rock evidence supports the requested style; coherent alternative plus rock/indie-rock tags also qualify. |
| T031: Create/Destroy | Art vs Science | 0 → 0 | Electronic/hip-hop mashup, indie pop, or electronic metadata does not substantiate the requested alternative-rock style. |
| T055: rock + roll | EDEN | 2 → 2 | Alternative-rock evidence supports the requested style; coherent alternative plus rock/indie-rock tags also qualify. Review caveat: Alternative Rock is explicit. Sparse metadata is not grounds for treating a broad rock request as irrelevant, but cannot prove seed or mood similarity. |

## tobi-g02: Recommend jazz music

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T095: Take It Or Leave It | Jet | 0 → 0 | The metadata does not provide credible jazz evidence for this request. |
| T078: Double Pump | Girl Talk | 0 → 0 | The metadata does not provide credible jazz evidence for this request. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T188: Destiny | The Jacksons | 0 → 0 | The metadata does not provide credible jazz evidence for this request. |
| T157: Any Advice? | Nonpoint | 0 → 0 | The metadata does not provide credible jazz evidence for this request. |
| T172: Barrytown | Steely Dan | 2 → 1 | Jazz-rock or jazz-hop is a defensible partial jazz match; the dominant crossover style warrants less than full credit. |
| T015: Get A Hold | A Tribe Called Quest | 0 → 1 | Jazz-rock or jazz-hop is a defensible partial jazz match; the dominant crossover style warrants less than full credit. |
| T080: Chloe | GROUPLOVE, Grouplove | 0 → 0 | The metadata does not provide credible jazz evidence for this request. |
| T087: Doubts 2 | Ibrahim Maalouf | 2 → 2 | Multiple jazz/subgenre tags establish a strong match to the broad jazz request. |
| T056: Uh Oh Hello | Elefant | 0 → 0 | The metadata does not provide credible jazz evidence for this request. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 0 → 0 | The metadata does not provide credible jazz evidence for this request. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 0 → 0 | The metadata does not provide credible jazz evidence for this request. |
| T065: Any Party | Feist | 0 → 0 | The metadata does not provide credible jazz evidence for this request. |
| T134: Ángel de amor | Maná | 0 → 0 | The metadata does not provide credible jazz evidence for this request. |
| T165: Getaway | RuPaul | 0 → 0 | The metadata does not provide credible jazz evidence for this request. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | The metadata does not provide credible jazz evidence for this request. |
| T156: Showbiz | Muse | 0 → 0 | The metadata does not provide credible jazz evidence for this request. Review caveat: Electronic is an isolated tag among dominant alternative/progressive-rock evidence; it does not justify strong Daft Punk similarity. |
| T181: Yoyo Waltz | The Cinematic Orchestra | 2 → 2 | Multiple jazz/subgenre tags establish a strong match to the broad jazz request. Review caveat: Jazz/downtempo dominates; the generic Rock tag warrants at most partial broad-rock relevance, not Numb-like relevance. |
| T176: Flim | The Bad Plus | 2 → 2 | Multiple jazz/subgenre tags establish a strong match to the broad jazz request. |
| T091: marshmallow | IU | 0 → 0 | The metadata does not provide credible jazz evidence for this request. Review caveat: Upbeat is inferred from happy, bubblegum pop and teen-pop tags; listening confirmation remains needed. |
| T163: Get Right | Pearl Jam | 0 → 0 | The metadata does not provide credible jazz evidence for this request. |

## tobi-g03: Recommend hip hop music

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T192: Devil Music | The Pharcyde | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T100: JoJo's Chillin | Killer Mike | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T078: Double Pump | Girl Talk | 2 → 1 | The mashup includes hip-hop evidence but also substantial electronic/rock content; this supports a partial rather than unqualified genre match. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T098: DUCKWORTH. | Kendrick Lamar | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. Review caveat: Trap rap occurs with conscious/west-coast hip-hop tags, but the actual 21 Savage Numb seed has no trap tag. A shared seed style is not verified; feedback relevance is partial pending listening. |
| T013: Like My Style | 50 Cent | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T205: Get Sum | Wiz Khalifa | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T101: R.A.P. Music | Killer Mike | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T151: Get Ur Freak On | Missy Elliott | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T057: 3 a.m. | Eminem | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T141: Get Right Witcha | Migos | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. Review caveat: Migos has trap and hip-hop tags. The 21 Savage Numb seed has only Hip-Hop/Rap and artist tags; a specific shared trap style cannot be asserted from the seed record. |
| T063: President Heartbeat | Everything Everything | 0 → 0 | Alternative/pop/rock evidence, without hip-hop evidence, does not meet the request. |
| T040: Pop's Rap | Common | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T015: Get A Hold | A Tribe Called Quest | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T012: Get In My Car | 50 Cent | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T102: 808 Man | KMD | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T083: Take A Look (At Yourself) | Guru's Jazzmatazz, Roy Ayers, Guru | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T014: What Up Gangsta | 50 Cent | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T088: Reverse Pimpology | Immortal Technique | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T204: November | Tyler, The Creator | 2 → 2 | Hip-hop/rap and its documented subgenres directly satisfy the broad hip-hop request. |
| T085: MMMBop | Hanson | 0 → 0 | Alternative/pop/rock evidence, without hip-hop evidence, does not meet the request. |

## tobi-g04: Recommend upbeat pop music

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T078: Double Pump | Girl Talk | 1 → 1 | Only one requested aspect is well supported: upbeat crossover without pop, or pop without convincing upbeat evidence. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T069: Poppin' Tags | Future | 0 → 0 | Rap/hip-hop tags or a title mentioning pop do not establish upbeat pop. |
| T081: Pop Music | Gucci Mane | 0 → 0 | Rap/hip-hop tags or a title mentioning pop do not establish upbeat pop. |
| T057: 3 a.m. | Eminem | 0 → 0 | Rap/hip-hop tags or a title mentioning pop do not establish upbeat pop. |
| T080: Chloe | GROUPLOVE, Grouplove | 1 → 2 | Pop/indie-pop/Latin-pop evidence occurs with upbeat, uplifting, happy, or dance-oriented tags; both requested aspects are supported, with inferred mood cases flagged. |
| T056: Uh Oh Hello | Elefant | 1 → 2 | Pop/indie-pop/Latin-pop evidence occurs with upbeat, uplifting, happy, or dance-oriented tags; both requested aspects are supported, with inferred mood cases flagged. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 1 → 1 | Only one requested aspect is well supported: upbeat crossover without pop, or pop without convincing upbeat evidence. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T129: Extraordinary | Liz Phair | 2 → 2 | Pop/indie-pop/Latin-pop evidence occurs with upbeat, uplifting, happy, or dance-oriented tags; both requested aspects are supported, with inferred mood cases flagged. |
| T158: Television, Television | OK Go | 2 → 2 | Pop/indie-pop/Latin-pop evidence occurs with upbeat, uplifting, happy, or dance-oriented tags; both requested aspects are supported, with inferred mood cases flagged. |
| T041: Get It Anyway | Cypress Hill | 0 → 0 | Rap/hip-hop tags or a title mentioning pop do not establish upbeat pop. |
| T053: Pop Style | Drake | 0 → 0 | Rap/hip-hop tags or a title mentioning pop do not establish upbeat pop. |
| T134: Ángel de amor | Maná | 1 → 2 | Pop/indie-pop/Latin-pop evidence occurs with upbeat, uplifting, happy, or dance-oriented tags; both requested aspects are supported, with inferred mood cases flagged. |
| T169: Carolina | Seu Jorge | 2 → 2 | Pop/indie-pop/Latin-pop evidence occurs with upbeat, uplifting, happy, or dance-oriented tags; both requested aspects are supported, with inferred mood cases flagged. |
| T025: 3WW | alt-J | 0 → 1 | Only one requested aspect is well supported: upbeat crossover without pop, or pop without convincing upbeat evidence. Review caveat: Alternative Rock is supplied, but ambient/art-pop tags also occur; genre classification is metadata-based. |
| T164: Gettin Up | Q-Tip | 1 → 1 | Only one requested aspect is well supported: upbeat crossover without pop, or pop without convincing upbeat evidence. |
| T091: marshmallow | IU | 1 → 2 | Pop/indie-pop/Latin-pop evidence occurs with upbeat, uplifting, happy, or dance-oriented tags; both requested aspects are supported, with inferred mood cases flagged. Review caveat: Upbeat is inferred from happy, bubblegum pop and teen-pop tags; listening confirmation remains needed. |
| T139: Shame | Matchbox Twenty, Archilla, Matt | 1 → 1 | Only one requested aspect is well supported: upbeat crossover without pop, or pop without convincing upbeat evidence. |
| T052: 0 To 100 / The Catch Up | Drake | 0 → 0 | Rap/hip-hop tags or a title mentioning pop do not establish upbeat pop. |
| T200: Morphine | Tony Moran, Michael Jackson | 1 → 1 | Only one requested aspect is well supported: upbeat crossover without pop, or pop without convincing upbeat evidence. |
| T138: Get Up | Mary Mary | 2 → 2 | Pop/indie-pop/Latin-pop evidence occurs with upbeat, uplifting, happy, or dance-oriented tags; both requested aspects are supported, with inferred mood cases flagged. Review caveat: Gospel/R&B/pop with uplifting and dance evidence is not automatically mellow R&B. |

## tobi-g05: Recommend mellow R&B music

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T193: Brightest Hour | The Submarines | 1 → 1 | Mellow/chill evidence without R&B, or R&B/soul evidence without a convincing mellow mood, meets only part of the request. Review caveat: Skyfall similarity is an AI interpretation of piano/female-vocal/sad/calm/indie-pop tags, not an audio comparison or independently verified similarity fact. |
| T078: Double Pump | Girl Talk | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T074: Relax, Relapse | Get Scared | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. |
| T157: Any Advice? | Nonpoint | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. |
| T057: 3 a.m. | Eminem | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. |
| T080: Chloe | GROUPLOVE, Grouplove | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. |
| T174: Little Bird | Steve Tannen, Deb Talan, The Weepies | 1 → 1 | Mellow/chill evidence without R&B, or R&B/soul evidence without a convincing mellow mood, meets only part of the request. |
| T056: Uh Oh Hello | Elefant | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 1 → 1 | Mellow/chill evidence without R&B, or R&B/soul evidence without a convincing mellow mood, meets only part of the request. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. |
| T134: Ángel de amor | Maná | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. |
| T104: Failure | Laura Marling | 1 → 1 | Mellow/chill evidence without R&B, or R&B/soul evidence without a convincing mellow mood, meets only part of the request. Review caveat: Folk/acoustic/singer-songwriter evidence dominates despite folk-rock/pop/rock tags; contextual relevance is partial. |
| T140: 3am | Meghan Trainor | 1 → 1 | Mellow/chill evidence without R&B, or R&B/soul evidence without a convincing mellow mood, meets only part of the request. |
| T154: Rock N Roll | Mos Def | 0 → 1 | Mellow/chill evidence without R&B, or R&B/soul evidence without a convincing mellow mood, meets only part of the request. |
| T178: My Tiger My Heart | The Boy Least Likely To | 1 → 1 | Mellow/chill evidence without R&B, or R&B/soul evidence without a convincing mellow mood, meets only part of the request. |
| T091: marshmallow | IU | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. Review caveat: Upbeat is inferred from happy, bubblegum pop and teen-pop tags; listening confirmation remains needed. |
| T075: Suffer | Get Scared | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. |
| T208: Get Started | Young Fathers | 0 → 0 | The available tags do not substantiate mellow R&B; incidental chill/mashup evidence is too weak to establish the requested musical combination. Review caveat: Only Alternative is supplied; do not infer rock or hip-hop from that label alone. |
| T138: Get Up | Mary Mary | 1 → 1 | Mellow/chill evidence without R&B, or R&B/soul evidence without a convincing mellow mood, meets only part of the request. Review caveat: Gospel/R&B/pop with uplifting and dance evidence is not automatically mellow R&B. |

## tobi-a01: Recommend music similar to Linkin Park

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T175: Get Loud | STS9 | 0 → 0 | Electronic/dance or R&B-only metadata does not substantiate similarity to the documented Linkin Park style. |
| T157: Any Advice? | Nonpoint | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |
| T118: In the End, In The End | Linkin Park | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |
| T143: P5hng Me Aw*y | Mike Shinoda, Stephen Richards, Linkin Park | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |
| T097: Get Like | Kehlani | 0 → 0 | Electronic/dance or R&B-only metadata does not substantiate similarity to the documented Linkin Park style. Review caveat: R&B/Soul is valid genre evidence but does not prove a mellow mood or similarity to Linkin Park, 21 Savage, or Adele. |
| T120: Not Alone | Linkin Park | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |
| T162: Options | PARTYNEXTDOOR | 0 → 0 | Electronic/dance or R&B-only metadata does not substantiate similarity to the documented Linkin Park style. Review caveat: R&B/Soul is meaningful but sparse. It is not evidence of a specific mood, hip-hop/trap style, or close seed similarity. |
| T201: Getaway | Train | 2 → 1 | Soft/alternative/hard rock supplies broad relatedness, but lacks the specific nu-metal/alternative-metal connection supporting a strong similarity claim. |
| T117: H! Vltg3 (Evidence Reanimation) [feat. Pharoahe Monch and DJ Babu], H! Vltg3 | Linkin Park | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |
| T179: Get Up On It Like This | The Chemical Brothers | 0 → 0 | Electronic/dance or R&B-only metadata does not substantiate similarity to the documented Linkin Park style. |
| T109: Breaking The Habit - Live At Milton Keynes | Linkin Park | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |
| T165: Getaway | RuPaul | 0 → 0 | Electronic/dance or R&B-only metadata does not substantiate similarity to the documented Linkin Park style. |
| T161: All For Nothing (feat. Page Hamilton) | Page Hamilton, Linkin Park | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | Electronic/dance or R&B-only metadata does not substantiate similarity to the documented Linkin Park style. |
| T115: Forgotten | Linkin Park | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |
| T166: Getaway | Saint Motel | 2 → 1 | Soft/alternative/hard rock supplies broad relatedness, but lacks the specific nu-metal/alternative-metal connection supporting a strong similarity claim. |
| T124: Runaway - Live In Texas | Linkin Park | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |
| T126: Shadow Of The Day - Live At Milton Keynes | Linkin Park | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |
| T016: Get It Hot | AC/DC | 2 → 1 | Soft/alternative/hard rock supplies broad relatedness, but lacks the specific nu-metal/alternative-metal connection supporting a strong similarity claim. |
| T113: Faint | Linkin Park | 2 → 2 | The candidate is verified Linkin Park music or has closely matching nu-metal/alternative-rock evidence. |

## tobi-a02: Recommend music similar to 21 Savage

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T207: Gucci On My (feat. 21 Savage, YG & Migos) | YG, 21 Savage, Migos, Mike WiLL Made-It | 2 → 2 | 21 Savage is a verified credited artist and hip-hop/rap music is documented, including sparsely tagged records and collaborations. |
| T157: Any Advice? | Nonpoint | 0 → 0 | No 21 Savage credit or defensible matching hip-hop/trap evidence is supplied by this record; adjacent R&B alone is insufficient here. |
| T097: Get Like | Kehlani | 0 → 0 | No 21 Savage credit or defensible matching hip-hop/trap evidence is supplied by this record; adjacent R&B alone is insufficient here. Review caveat: R&B/Soul is valid genre evidence but does not prove a mellow mood or similarity to Linkin Park, 21 Savage, or Adele. |
| T162: Options | PARTYNEXTDOOR | 0 → 0 | No 21 Savage credit or defensible matching hip-hop/trap evidence is supplied by this record; adjacent R&B alone is insufficient here. Review caveat: R&B/Soul is meaningful but sparse. It is not evidence of a specific mood, hip-hop/trap style, or close seed similarity. |
| T006: Numb | 21 Savage | 2 → 2 | 21 Savage is a verified credited artist and hip-hop/rap music is documented, including sparsely tagged records and collaborations. Review caveat: Sparse Hip-Hop/Rap tag is valid; do not infer irrelevant because the exact alias is absent from an application helper. |
| T005: Mad High | 21 Savage | 2 → 2 | 21 Savage is a verified credited artist and hip-hop/rap music is documented, including sparsely tagged records and collaborations. |
| T004: Famous | 21 Savage | 2 → 2 | 21 Savage is a verified credited artist and hip-hop/rap music is documented, including sparsely tagged records and collaborations. |
| T202: Getting Even | Trapt | 0 → 0 | No 21 Savage credit or defensible matching hip-hop/trap evidence is supplied by this record; adjacent R&B alone is insufficient here. Review caveat: Metal/Rock/Alternative are meaningful but too broad to prove close seed similarity. |
| T179: Get Up On It Like This | The Chemical Brothers | 0 → 0 | No 21 Savage credit or defensible matching hip-hop/trap evidence is supplied by this record; adjacent R&B alone is insufficient here. |
| T001: Bad Guy | 21 Savage | 2 → 2 | 21 Savage is a verified credited artist and hip-hop/rap music is documented, including sparsely tagged records and collaborations. |
| T165: Getaway | RuPaul | 0 → 0 | No 21 Savage credit or defensible matching hip-hop/trap evidence is supplied by this record; adjacent R&B alone is insufficient here. |
| T009: No Advance | 21 Savage, Metro Boomin | 2 → 2 | 21 Savage is a verified credited artist and hip-hop/rap music is documented, including sparsely tagged records and collaborations. |
| T076: Voodoo | Get Scared | 0 → 0 | No 21 Savage credit or defensible matching hip-hop/trap evidence is supplied by this record; adjacent R&B alone is insufficient here. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | No 21 Savage credit or defensible matching hip-hop/trap evidence is supplied by this record; adjacent R&B alone is insufficient here. |
| T077: Whore | Get Scared | 0 → 0 | No 21 Savage credit or defensible matching hip-hop/trap evidence is supplied by this record; adjacent R&B alone is insufficient here. |
| T007: Special | 21 Savage | 2 → 2 | 21 Savage is a verified credited artist and hip-hop/rap music is documented, including sparsely tagged records and collaborations. |
| T008: Feel It | 21 Savage, Metro Boomin | 2 → 2 | 21 Savage is a verified credited artist and hip-hop/rap music is documented, including sparsely tagged records and collaborations. |
| T011: Real Nigga | 21 Savage, Metro Boomin | 2 → 2 | 21 Savage is a verified credited artist and hip-hop/rap music is documented, including sparsely tagged records and collaborations. |
| T003: FaceTime | 21 Savage | 2 → 2 | 21 Savage is a verified credited artist and hip-hop/rap music is documented, including sparsely tagged records and collaborations. |
| T016: Get It Hot | AC/DC | 0 → 0 | No 21 Savage credit or defensible matching hip-hop/trap evidence is supplied by this record; adjacent R&B alone is insufficient here. |

## tobi-a03: Recommend music similar to Adele

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T066: Get Not High, Get Not Low | Feist | 2 → 1 | Pop/soul/R&B or a documented piano reinterpretation provides partial relatedness, but the record does not establish a close match to Adele's documented soul/pop ballad style. |
| T019: Skyfall - Full Length | Adele | 2 → 2 | This is verified Adele music with pop, soul, ballad, and vocal evidence. |
| T060: Hello | Erato | 2 → 1 | Pop/soul/R&B or a documented piano reinterpretation provides partial relatedness, but the record does not establish a close match to Adele's documented soul/pop ballad style. |
| T157: Any Advice? | Nonpoint | 0 → 0 | Dominant lo-fi/alternative guitar rock, punk, electronic music, jazz covers, or an uninformative record does not substantiate Adele similarity. An artist named Robyn Adele Anderson is not Adele. |
| T097: Get Like | Kehlani | 0 → 1 | Pop/soul/R&B or a documented piano reinterpretation provides partial relatedness, but the record does not establish a close match to Adele's documented soul/pop ballad style. Review caveat: R&B/Soul is valid genre evidence but does not prove a mellow mood or similarity to Linkin Park, 21 Savage, or Adele. |
| T080: Chloe | GROUPLOVE, Grouplove | 0 → 1 | Pop/soul/R&B or a documented piano reinterpretation provides partial relatedness, but the record does not establish a close match to Adele's documented soul/pop ballad style. |
| T082: Gold Star For Robot Boy | Guided By Voices | 2 → 0 | Dominant lo-fi/alternative guitar rock, punk, electronic music, jazz covers, or an uninformative record does not substantiate Adele similarity. An artist named Robyn Adele Anderson is not Adele. Review caveat: Pop/indie-pop tags coexist with lo-fi, raw alternative/indie-rock evidence. The record does not substantiate Adele-like soul/pop ballad style despite generic pop overlap. |
| T162: Options | PARTYNEXTDOOR | 0 → 1 | Pop/soul/R&B or a documented piano reinterpretation provides partial relatedness, but the record does not establish a close match to Adele's documented soul/pop ballad style. Review caveat: R&B/Soul is meaningful but sparse. It is not evidence of a specific mood, hip-hop/trap style, or close seed similarity. |
| T037: Get Mine, Get Yours | Christina Aguilera | 2 → 1 | Pop/soul/R&B or a documented piano reinterpretation provides partial relatedness, but the record does not establish a close match to Adele's documented soul/pop ballad style. |
| T130: Get Some | Lykke Li | 2 → 1 | Pop/soul/R&B or a documented piano reinterpretation provides partial relatedness, but the record does not establish a close match to Adele's documented soul/pop ballad style. |
| T194: Darkest Things | The Submarines | 2 → 1 | Pop/soul/R&B or a documented piano reinterpretation provides partial relatedness, but the record does not establish a close match to Adele's documented soul/pop ballad style. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T092: The Boat Dreams from the Hill | Jawbreaker | 0 → 0 | Dominant lo-fi/alternative guitar rock, punk, electronic music, jazz covers, or an uninformative record does not substantiate Adele similarity. An artist named Robyn Adele Anderson is not Adele. |
| T020: Rolling in the Deep (vocal version) | Adele Adkins, The Piano Guys | 2 → 1 | Pop/soul/R&B or a documented piano reinterpretation provides partial relatedness, but the record does not establish a close match to Adele's documented soul/pop ballad style. Review caveat: Adele Adkins is a catalog credit alongside The Piano Guys; a piano/vocal reinterpretation does not automatically establish a close match to Adele's own recordings. |
| T165: Getaway | RuPaul | 0 → 0 | Dominant lo-fi/alternative guitar rock, punk, electronic music, jazz covers, or an uninformative record does not substantiate Adele similarity. An artist named Robyn Adele Anderson is not Adele. |
| T099: Get Em | Kevin Gates | 0 → 0 | Dominant lo-fi/alternative guitar rock, punk, electronic music, jazz covers, or an uninformative record does not substantiate Adele similarity. An artist named Robyn Adele Anderson is not Adele. Review caveat: Only Others tags are provided. No musical genre or similarity is invented from the artist identity. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | Dominant lo-fi/alternative guitar rock, punk, electronic music, jazz covers, or an uninformative record does not substantiate Adele similarity. An artist named Robyn Adele Anderson is not Adele. |
| T156: Showbiz | Muse | 0 → 0 | Dominant lo-fi/alternative guitar rock, punk, electronic music, jazz covers, or an uninformative record does not substantiate Adele similarity. An artist named Robyn Adele Anderson is not Adele. Review caveat: Electronic is an isolated tag among dominant alternative/progressive-rock evidence; it does not justify strong Daft Punk similarity. |
| T168: Timber | Scott Bradlee's Postmodern Jukebox, Robyn Adele Anderson | 0 → 0 | Dominant lo-fi/alternative guitar rock, punk, electronic music, jazz covers, or an uninformative record does not substantiate Adele similarity. An artist named Robyn Adele Anderson is not Adele. Review caveat: Robyn Adele Anderson is a different credited identity from Adele. Jazz cover metadata does not establish Adele similarity. |
| T139: Shame | Matchbox Twenty, Archilla, Matt | 2 → 1 | Pop/soul/R&B or a documented piano reinterpretation provides partial relatedness, but the record does not establish a close match to Adele's documented soul/pop ballad style. |
| T167: Gangsta's Paradise | Scott Bradlee's Postmodern Jukebox, Robyn Adele Anderson | 0 → 0 | Dominant lo-fi/alternative guitar rock, punk, electronic music, jazz covers, or an uninformative record does not substantiate Adele similarity. An artist named Robyn Adele Anderson is not Adele. Review caveat: Robyn Adele Anderson is a different credited identity from Adele. Jazz cover metadata does not establish Adele similarity. |

## tobi-a04: Recommend music similar to Miles Davis

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T137: Straight, No Chaser | Mark Wilder, Miles Davis | 2 → 2 | Miles Davis is explicitly credited and jazz/trumpet/bop/fusion metadata verifies the musical relationship. |
| T066: Get Not High, Get Not Low | Feist | 0 → 0 | The record supplies no defensible jazz/Miles Davis relationship. |
| T205: Get Sum | Wiz Khalifa | 0 → 0 | The record supplies no defensible jazz/Miles Davis relationship. |
| T157: Any Advice? | Nonpoint | 0 → 0 | The record supplies no defensible jazz/Miles Davis relationship. |
| T149: So What, So What (feat. John Coltrane, Cannonball Adderley & Bill Evans) | Miles Davis | 2 → 2 | Miles Davis is explicitly credited and jazz/trumpet/bop/fusion metadata verifies the musical relationship. |
| T150: Sweet Sue, Just You | Miles Davis | 2 → 2 | Miles Davis is explicitly credited and jazz/trumpet/bop/fusion metadata verifies the musical relationship. |
| T015: Get A Hold | A Tribe Called Quest | 0 → 1 | Jazz-hop and jazz-influenced hip-hop are adjacent to jazz, but do not establish a close match to Miles Davis recordings. |
| T147: On Green Dolphin Street (feat. John Coltrane, Cannonball Adderley & Bill Evans), On Green Dolphin Street | Miles Davis | 2 → 2 | Miles Davis is explicitly credited and jazz/trumpet/bop/fusion metadata verifies the musical relationship. |
| T145: Miles Ahead | Miles Davis | 2 → 2 | Miles Davis is explicitly credited and jazz/trumpet/bop/fusion metadata verifies the musical relationship. |
| T162: Options | PARTYNEXTDOOR | 0 → 0 | The record supplies no defensible jazz/Miles Davis relationship. Review caveat: R&B/Soul is meaningful but sparse. It is not evidence of a specific mood, hip-hop/trap style, or close seed similarity. |
| T144: 'Round Midnight (feat. John Coltrane, Red Garland, Paul Chambers & Philly Joe Jones), 'Round Midnight | Miles Davis | 2 → 2 | Miles Davis is explicitly credited and jazz/trumpet/bop/fusion metadata verifies the musical relationship. Review caveat: The catalog combines title variants and broad tags. Miles Davis and jazz are verified, but the exact recording/version is not adjudicated. |
| T202: Getting Even | Trapt | 0 → 0 | The record supplies no defensible jazz/Miles Davis relationship. Review caveat: Metal/Rock/Alternative are meaningful but too broad to prove close seed similarity. |
| T148: On the Corner / New York Girl / Thinkin' of one Thing and Doin' Another / Vote for Miles, On the Corner / New York Girl / Thinkin' of One Thing and Doin' Another / Vote for Miles | Miles Davis | 2 → 2 | Miles Davis is explicitly credited and jazz/trumpet/bop/fusion metadata verifies the musical relationship. |
| T065: Any Party | Feist | 0 → 0 | The record supplies no defensible jazz/Miles Davis relationship. |
| T165: Getaway | RuPaul | 0 → 0 | The record supplies no defensible jazz/Miles Davis relationship. |
| T099: Get Em | Kevin Gates | 0 → 0 | The record supplies no defensible jazz/Miles Davis relationship. Review caveat: Only Others tags are provided. No musical genre or similarity is invented from the artist identity. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | The record supplies no defensible jazz/Miles Davis relationship. |
| T136: Stella By Starlight, Stella By Starlight (feat. John Coltrane & Bill Evans) | Mark Wilder, Miles Davis | 2 → 2 | Miles Davis is explicitly credited and jazz/trumpet/bop/fusion metadata verifies the musical relationship. |
| T135: Sid's Ahead | Mark Wilder, Miles Davis | 2 → 2 | Miles Davis is explicitly credited and jazz/trumpet/bop/fusion metadata verifies the musical relationship. |
| T146: Milestones, Milestones (feat. John Coltrane, Cannonball Adderley, Red Garland, Paul Chambers & Philly Joe Jones) | Miles Davis | 2 → 2 | Miles Davis is explicitly credited and jazz/trumpet/bop/fusion metadata verifies the musical relationship. |

## tobi-a05: Recommend music similar to Daft Punk

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T047: Revolution 909 | Daft Punk | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. |
| T058: Artists vs Turtles | Epic Rap Battles of History | 0 → 0 | The available music metadata does not substantiate Daft Punk similarity. A title resembling an episode with only Others tags is insufficient evidence of a relevant song. |
| T044: Digital Love | Daft Punk | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. |
| T157: Any Advice? | Nonpoint | 0 → 0 | The available music metadata does not substantiate Daft Punk similarity. A title resembling an episode with only Others tags is insufficient evidence of a relevant song. |
| T050: TRON Legacy (End Titles) | Daft Punk | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. Review caveat: Daft Punk and electronic are verified; soundtrack-oriented material may not match an assessor's intended dance/house interpretation. |
| T048: The Prime Time Of Your Life | Daft Punk | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. |
| T049: Too Long (Gonzales Version) | Daft Punk | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. Review caveat: Daft Punk is credited, but Gonzales Version plus piano/acoustic/mellow tags may differ substantially from their electronic dance recordings. |
| T162: Options | PARTYNEXTDOOR | 0 → 0 | The available music metadata does not substantiate Daft Punk similarity. A title resembling an episode with only Others tags is insufficient evidence of a relevant song. Review caveat: R&B/Soul is meaningful but sparse. It is not evidence of a specific mood, hip-hop/trap style, or close seed similarity. |
| T201: Getaway | Train | 0 → 0 | The available music metadata does not substantiate Daft Punk similarity. A title resembling an episode with only Others tags is insufficient evidence of a relevant song. |
| T202: Getting Even | Trapt | 0 → 0 | The available music metadata does not substantiate Daft Punk similarity. A title resembling an episode with only Others tags is insufficient evidence of a relevant song. Review caveat: Metal/Rock/Alternative are meaningful but too broad to prove close seed similarity. |
| T179: Get Up On It Like This | The Chemical Brothers | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. |
| T042: Aerodynamic (Daft Punk Remix) | Daft Punk | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. |
| T165: Getaway | RuPaul | 2 → 1 | Dance/electronic tags support broad relatedness, but no more specific Daft Punk/house connection is verified. |
| T099: Get Em | Kevin Gates | 0 → 0 | The available music metadata does not substantiate Daft Punk similarity. A title resembling an episode with only Others tags is insufficient evidence of a relevant song. Review caveat: Only Others tags are provided. No musical genre or similarity is invented from the artist identity. |
| T180: Music : Response | The Chemical Brothers | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. |
| T156: Showbiz | Muse | 2 → 0 | The available music metadata does not substantiate Daft Punk similarity. A title resembling an episode with only Others tags is insufficient evidence of a relevant song. Review caveat: Electronic is an isolated tag among dominant alternative/progressive-rock evidence; it does not justify strong Daft Punk similarity. |
| T046: Phoenix (Basement Jaxx Remix) | Daft Punk | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. |
| T043: Crescendolls | Daft Punk | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. |
| T045: Episode 2, Todd Edwards | Daft Punk | 2 → 0 | The available music metadata does not substantiate Daft Punk similarity. A title resembling an episode with only Others tags is insufficient evidence of a relevant song. Review caveat: Episode 2, Todd Edwards is tagged only Others. Treat as insufficient evidence of relevant music; do not assert that it is definitely spoken audio without listening. |
| T106: Daft Punk Is Playing At My House, Daft Punk Is Playing at My House | LCD Soundsystem | 2 → 2 | Verified Daft Punk music, or overlapping electronic dance/house/techno styles, supports the requested artist relationship; soundtrack and acoustic reinterpretations are flagged for style uncertainty. Review caveat: The title names Daft Punk, but the similarity label relies on house/disco/electronic/dance evidence, not that title. |

## tobi-p01: Recommend more songs that fit this rock playlist

Context: `{"playlist": [{"title": "Numb", "artist": "Linkin Park"}], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T175: Get Loud | STS9 | 0 → 0 | Electronic/R&B or generic Alternative tags do not substantiate compatibility with the documented rock/nu-metal playlist seed. |
| T118: In the End, In The End | Linkin Park | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T097: Get Like | Kehlani | 0 → 0 | Electronic/R&B or generic Alternative tags do not substantiate compatibility with the documented rock/nu-metal playlist seed. Review caveat: R&B/Soul is valid genre evidence but does not prove a mellow mood or similarity to Linkin Park, 21 Savage, or Adele. |
| T128: With You | Linkin Park | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T112: Easier To Run | Linkin Park | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T162: Options | PARTYNEXTDOOR | 0 → 0 | Electronic/R&B or generic Alternative tags do not substantiate compatibility with the documented rock/nu-metal playlist seed. Review caveat: R&B/Soul is meaningful but sparse. It is not evidence of a specific mood, hip-hop/trap style, or close seed similarity. |
| T201: Getaway | Train | 2 → 1 | Soft alternative rock or punk/post-hardcore has broad rock overlap but weaker specific similarity to the Numb seed. |
| T028: Get Up and Move On | Alvarez Kings | 0 → 0 | Electronic/R&B or generic Alternative tags do not substantiate compatibility with the documented rock/nu-metal playlist seed. |
| T127: Somewhere I Belong | Linkin Park | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T122: Papercut | Linkin Park | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T165: Getaway | RuPaul | 0 → 0 | Electronic/R&B or generic Alternative tags do not substantiate compatibility with the documented rock/nu-metal playlist seed. |
| T068: Suggestion | Fugazi | 2 → 1 | Soft alternative rock or punk/post-hardcore has broad rock overlap but weaker specific similarity to the Numb seed. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | Electronic/R&B or generic Alternative tags do not substantiate compatibility with the documented rock/nu-metal playlist seed. |
| T115: Forgotten | Linkin Park | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T111: Crawling | Linkin Park | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T071: Hate | Get Scared | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T114: Figure.09 | Linkin Park | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T116: From The Inside | Linkin Park | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T090: Personality Cult | Issues | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |
| T113: Faint | Linkin Park | 2 → 2 | Linkin Park credit or specific alternative-metal/nu-metal plus rock evidence is compatible with the Numb playlist seed. |

## tobi-p02: Recommend more hip hop for my playlist

Context: `{"playlist": [{"title": "Numb", "artist": "21 Savage"}], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T207: Gucci On My (feat. 21 Savage, YG & Migos) | YG, 21 Savage, Migos, Mike WiLL Made-It | 2 → 2 | A verified 21 Savage credit and hip-hop evidence support the playlist's artist and genre; Hip-Hop/Rap is valid evidence even in sparse records. |
| T205: Get Sum | Wiz Khalifa | 2 → 1 | Hip-hop is verified, but the actual Numb seed has only Hip-Hop/Rap and artist tags; close track-level similarity or a shared trap style cannot be verified from that seed metadata. |
| T010: No Heart | 21 Savage, Metro Boomin | 2 → 2 | A verified 21 Savage credit and hip-hop evidence support the playlist's artist and genre; Hip-Hop/Rap is valid evidence even in sparse records. |
| T157: Any Advice? | Nonpoint | 0 → 0 | The supplied record does not substantiate hip-hop compatibility with the playlist seed. |
| T141: Get Right Witcha | Migos | 2 → 1 | Hip-hop is verified, but the actual Numb seed has only Hip-Hop/Rap and artist tags; close track-level similarity or a shared trap style cannot be verified from that seed metadata. Review caveat: Migos has trap and hip-hop tags. The 21 Savage Numb seed has only Hip-Hop/Rap and artist tags; a specific shared trap style cannot be asserted from the seed record. |
| T162: Options | PARTYNEXTDOOR | 0 → 0 | The supplied record does not substantiate hip-hop compatibility with the playlist seed. Review caveat: R&B/Soul is meaningful but sparse. It is not evidence of a specific mood, hip-hop/trap style, or close seed similarity. |
| T201: Getaway | Train | 0 → 0 | The supplied record does not substantiate hip-hop compatibility with the playlist seed. |
| T005: Mad High | 21 Savage | 2 → 2 | A verified 21 Savage credit and hip-hop evidence support the playlist's artist and genre; Hip-Hop/Rap is valid evidence even in sparse records. |
| T004: Famous | 21 Savage | 2 → 2 | A verified 21 Savage credit and hip-hop evidence support the playlist's artist and genre; Hip-Hop/Rap is valid evidence even in sparse records. |
| T202: Getting Even | Trapt | 0 → 0 | The supplied record does not substantiate hip-hop compatibility with the playlist seed. Review caveat: Metal/Rock/Alternative are meaningful but too broad to prove close seed similarity. |
| T001: Bad Guy | 21 Savage | 2 → 2 | A verified 21 Savage credit and hip-hop evidence support the playlist's artist and genre; Hip-Hop/Rap is valid evidence even in sparse records. |
| T165: Getaway | RuPaul | 0 → 0 | The supplied record does not substantiate hip-hop compatibility with the playlist seed. |
| T009: No Advance | 21 Savage, Metro Boomin | 2 → 2 | A verified 21 Savage credit and hip-hop evidence support the playlist's artist and genre; Hip-Hop/Rap is valid evidence even in sparse records. |
| T002: Close My Eyes | 21 Savage | 2 → 2 | A verified 21 Savage credit and hip-hop evidence support the playlist's artist and genre; Hip-Hop/Rap is valid evidence even in sparse records. |
| T156: Showbiz | Muse | 0 → 0 | The supplied record does not substantiate hip-hop compatibility with the playlist seed. Review caveat: Electronic is an isolated tag among dominant alternative/progressive-rock evidence; it does not justify strong Daft Punk similarity. |
| T062: Fortune 500 | Everything Everything | 0 → 0 | The supplied record does not substantiate hip-hop compatibility with the playlist seed. |
| T007: Special | 21 Savage | 2 → 2 | A verified 21 Savage credit and hip-hop evidence support the playlist's artist and genre; Hip-Hop/Rap is valid evidence even in sparse records. |
| T008: Feel It | 21 Savage, Metro Boomin | 2 → 2 | A verified 21 Savage credit and hip-hop evidence support the playlist's artist and genre; Hip-Hop/Rap is valid evidence even in sparse records. |
| T003: FaceTime | 21 Savage | 2 → 2 | A verified 21 Savage credit and hip-hop evidence support the playlist's artist and genre; Hip-Hop/Rap is valid evidence even in sparse records. |
| T208: Get Started | Young Fathers | 0 → 0 | The supplied record does not substantiate hip-hop compatibility with the playlist seed. Review caveat: Only Alternative is supplied; do not infer rock or hip-hop from that label alone. |

## tobi-p03: Recommend more pop for this playlist

Context: `{"playlist": [{"title": "Skyfall - Full Length", "artist": "Adele"}], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T038: Welcome | Christina Aguilera | 2 → 2 | Pop with ballad, soul, orchestral/cinematic, piano, or compatible vocal evidence provides a specific relationship to the Skyfall seed. |
| T103: 24 | Lana Del Rey | 2 → 2 | Pop with ballad, soul, orchestral/cinematic, piano, or compatible vocal evidence provides a specific relationship to the Skyfall seed. |
| T084: 2049 | Hans Zimmer, Benjamin Wallfisch | 0 → 0 | Shared soundtrack/film/title associations, classical scores, or comedy rap do not establish the requested pop playlist fit. |
| T026: In Cold Blood | alt-J | 2 → 1 | Broad pop/dream-pop, a dance-oriented Bond theme, or a hard-rock Skyfall cover supplies a partial relationship but not a close pop-ballad style match. Review caveat: Dominant alternative/indie-rock metadata; the generic Pop tag gives only partial Skyfall playlist compatibility. |
| T094: Night Light | Jessie Ware | 2 → 2 | Pop with ballad, soul, orchestral/cinematic, piano, or compatible vocal evidence provides a specific relationship to the Skyfall seed. |
| T170: Auditions - From "Sing" Original Motion Picture Soundtrack | Sing Cast | 0 → 0 | Shared soundtrack/film/title associations, classical scores, or comedy rap do not establish the requested pop playlist fit. Review caveat: Auditions is soundtrack-tagged, with no verified pop style; do not assume every soundtrack excerpt fits the Skyfall playlist. |
| T203: Returns a King - Based Upon Themes by Elliot Goldenthal, Returns A King - Based Upon Themes by Elliot Goldenthal | Tyler Bates | 0 → 0 | Shared soundtrack/film/title associations, classical scores, or comedy rap do not establish the requested pop playlist fit. |
| T197: The Bloody Shot | Thomas Newman | 0 → 0 | Shared soundtrack/film/title associations, classical scores, or comedy rap do not establish the requested pop playlist fit. Review caveat: A Skyfall/Bond association and isolated Rock tag do not establish pop relevance. |
| T195: Breadcrumbs | Thomas Newman | 0 → 0 | Shared soundtrack/film/title associations, classical scores, or comedy rap do not establish the requested pop playlist fit. Review caveat: A Skyfall/Bond association alone does not establish pop relevance. |
| T199: The Moors | Thomas Newman | 0 → 0 | Shared soundtrack/film/title associations, classical scores, or comedy rap do not establish the requested pop playlist fit. Review caveat: A Skyfall/Bond association and classical/Rock tags do not establish pop relevance. |
| T034: All | Blackbird Blackbird | 0 → 1 | Broad pop/dream-pop, a dance-oriented Bond theme, or a hard-rock Skyfall cover supplies a partial relationship but not a close pop-ballad style match. |
| T132: Die Another Day | Madonna | 2 → 1 | Broad pop/dream-pop, a dance-oriented Bond theme, or a hard-rock Skyfall cover supplies a partial relationship but not a close pop-ballad style match. Review caveat: A shared Bond association is supported, but the dance/electropop style differs from the cinematic soul-ballad seed. |
| T196: Severine | Thomas Newman | 0 → 0 | Shared soundtrack/film/title associations, classical scores, or comedy rap do not establish the requested pop playlist fit. Review caveat: A Skyfall/Bond association alone does not establish pop relevance. |
| T198: The Chimera | Thomas Newman | 0 → 0 | Shared soundtrack/film/title associations, classical scores, or comedy rap do not establish the requested pop playlist fit. Review caveat: A Skyfall/Bond association alone does not establish pop relevance. |
| T039: A Thousand Years | Christina Perri | 2 → 2 | Pop with ballad, soul, orchestral/cinematic, piano, or compatible vocal evidence provides a specific relationship to the Skyfall seed. |
| T079: Human | Goldfrapp | 2 → 2 | Pop with ballad, soul, orchestral/cinematic, piano, or compatible vocal evidence provides a specific relationship to the Skyfall seed. Review caveat: Cinematic, strings, james bond, female vocals and art pop support a Skyfall relationship; no claim is made that the recordings share an actual soundtrack. |
| T159: Skyfall | Our Last Night | 0 → 1 | Broad pop/dream-pop, a dance-oriented Bond theme, or a hard-rock Skyfall cover supplies a partial relationship but not a close pop-ballad style match. Review caveat: A Skyfall cover with hard-rock tags has a composition/title relationship, but does not establish the requested pop playlist style. |
| T153: Schindler's List | Moldoni | 0 → 0 | Shared soundtrack/film/title associations, classical scores, or comedy rap do not establish the requested pop playlist fit. |
| T160: Try | P!nk | 2 → 2 | Pop with ballad, soul, orchestral/cinematic, piano, or compatible vocal evidence provides a specific relationship to the Skyfall seed. |
| T033: Skinny Love | Birdy | 2 → 2 | Pop with ballad, soul, orchestral/cinematic, piano, or compatible vocal evidence provides a specific relationship to the Skyfall seed. |
| T059: Steven Spielberg vs Alfred Hitchcock | Epic Rap Battles of History | 0 → 0 | Shared soundtrack/film/title associations, classical scores, or comedy rap do not establish the requested pop playlist fit. |

## tobi-p04: Recommend rock that fits both songs in my playlist

Context: `{"playlist": [{"title": "Numb", "artist": "Linkin Park"}, {"title": "In the End", "artist": "Linkin Park"}], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T175: Get Loud | STS9 | 0 → 0 | Electronic/dance or R&B metadata does not substantiate the requested two-seed rock relationship. |
| T128: With You | Linkin Park | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T162: Options | PARTYNEXTDOOR | 0 → 0 | Electronic/dance or R&B metadata does not substantiate the requested two-seed rock relationship. Review caveat: R&B/Soul is meaningful but sparse. It is not evidence of a specific mood, hip-hop/trap style, or close seed similarity. |
| T201: Getaway | Train | 2 → 1 | Soft/classic/indie rock or an instrumental electronic Linkin Park piece has only partial similarity to the two documented vocal rock seeds. |
| T110: By Myself | Linkin Park | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T024: Get Born Again | Alice In Chains | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T127: Somewhere I Belong | Linkin Park | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T187: Get It Together | The Go! Team | 0 → 1 | Soft/classic/indie rock or an instrumental electronic Linkin Park piece has only partial similarity to the two documented vocal rock seeds. |
| T122: Papercut | Linkin Park | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T165: Getaway | RuPaul | 0 → 0 | Electronic/dance or R&B metadata does not substantiate the requested two-seed rock relationship. |
| T115: Forgotten | Linkin Park | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T125: Session | Linkin Park | 2 → 1 | Soft/classic/indie rock or an instrumental electronic Linkin Park piece has only partial similarity to the two documented vocal rock seeds. Review caveat: Session is explicitly instrumental/electronic/breakbeat as well as rock-tagged. Same artist alone does not prove close similarity to both vocal rock seeds. |
| T111: Crawling | Linkin Park | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T185: Take It As It Comes | The Doors | 2 → 1 | Soft/classic/indie rock or an instrumental electronic Linkin Park piece has only partial similarity to the two documented vocal rock seeds. Review caveat: Classic/psychedelic rock dominates despite one alternative-rock tag; partial alternative-rock or Linkin Park similarity is more defensible than full credit. |
| T121: One Step Closer | Linkin Park | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T071: Hate | Get Scared | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T114: Figure.09 | Linkin Park | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T075: Suffer | Get Scared | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T123: Pushing Me Away | Linkin Park | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T090: Personality Cult | Issues | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |
| T113: Faint | Linkin Park | 2 → 2 | Verified Linkin Park rock or specific alternative-metal/nu-metal evidence supports compatibility with both Numb and In the End. |

## tobi-p05: Recommend music for my mixed rock and hip hop playlist

Context: `{"playlist": [{"title": "Numb", "artist": "Linkin Park"}, {"title": "Numb", "artist": "21 Savage"}], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T175: Get Loud | STS9 | 0 → 0 | Electronic/R&B or generic Alternative metadata alone does not substantiate either seed's requested rock/hip-hop relationship. |
| T119: Nobody's Listening | Linkin Park | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |
| T072: Mess | Get Scared | 2 → 1 | Alternative/emo/folk rock supplies broad overlap with one seed, but does not substantiate close nu-metal or 21 Savage hip-hop compatibility. |
| T157: Any Advice? | Nonpoint | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |
| T118: In the End, In The End | Linkin Park | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |
| T128: With You | Linkin Park | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |
| T162: Options | PARTYNEXTDOOR | 0 → 0 | Electronic/R&B or generic Alternative metadata alone does not substantiate either seed's requested rock/hip-hop relationship. Review caveat: R&B/Soul is meaningful but sparse. It is not evidence of a specific mood, hip-hop/trap style, or close seed similarity. |
| T202: Getting Even | Trapt | 2 → 1 | Alternative/emo/folk rock supplies broad overlap with one seed, but does not substantiate close nu-metal or 21 Savage hip-hop compatibility. Review caveat: Metal/Rock/Alternative are meaningful but too broad to prove close seed similarity. |
| T186: Career Day | The Format | 2 → 1 | Alternative/emo/folk rock supplies broad overlap with one seed, but does not substantiate close nu-metal or 21 Savage hip-hop compatibility. |
| T127: Somewhere I Belong | Linkin Park | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |
| T122: Papercut | Linkin Park | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |
| T165: Getaway | RuPaul | 0 → 0 | Electronic/R&B or generic Alternative metadata alone does not substantiate either seed's requested rock/hip-hop relationship. |
| T076: Voodoo | Get Scared | 2 → 1 | Alternative/emo/folk rock supplies broad overlap with one seed, but does not substantiate close nu-metal or 21 Savage hip-hop compatibility. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | Electronic/R&B or generic Alternative metadata alone does not substantiate either seed's requested rock/hip-hop relationship. |
| T115: Forgotten | Linkin Park | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |
| T111: Crawling | Linkin Park | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |
| T108: A Place For My Head | Linkin Park | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |
| T114: Figure.09 | Linkin Park | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |
| T208: Get Started | Young Fathers | 0 → 0 | Electronic/R&B or generic Alternative metadata alone does not substantiate either seed's requested rock/hip-hop relationship. Review caveat: Only Alternative is supplied; do not infer rock or hip-hop from that label alone. |
| T113: Faint | Linkin Park | 2 → 2 | Linkin Park credit or specific nu-metal evidence supports the rock seed of the mixed playlist. A candidate need not combine both seed genres. |

## tobi-e01: Recommend alternative rock but exclude Linkin Park

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": ["Linkin Park"], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T078: Double Pump | Girl Talk | 2 → 0 | The Linkin Park exclusion is satisfied, but electronic/mashup or indie-pop metadata does not substantiate alternative rock. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T027: Isolation | Alter Bridge | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. |
| T157: Any Advice? | Nonpoint | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. |
| T183: Revolution Rock | The Clash | 2 → 1 | Reggae/progressive/Latin/classic/hard rock is only a partial fit to alternative rock; the artist exclusion is satisfied. |
| T067: Argument | Fugazi | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. |
| T080: Chloe | GROUPLOVE, Grouplove | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. |
| T155: Falling Down | Muse | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. |
| T056: Uh Oh Hello | Elefant | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 2 → 0 | The Linkin Park exclusion is satisfied, but electronic/mashup or indie-pop metadata does not substantiate alternative rock. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 2 → 1 | Reggae/progressive/Latin/classic/hard rock is only a partial fit to alternative rock; the artist exclusion is satisfied. |
| T158: Television, Television | OK Go | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. |
| T134: Ángel de amor | Maná | 2 → 1 | Reggae/progressive/Latin/classic/hard rock is only a partial fit to alternative rock; the artist exclusion is satisfied. |
| T068: Suggestion | Fugazi | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. |
| T190: Love's A Game | The Magic Numbers | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. |
| T185: Take It As It Comes | The Doors | 2 → 1 | Reggae/progressive/Latin/classic/hard rock is only a partial fit to alternative rock; the artist exclusion is satisfied. Review caveat: Classic/psychedelic rock dominates despite one alternative-rock tag; partial alternative-rock or Linkin Park similarity is more defensible than full credit. |
| T182: Career Opportunities | The Clash | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. |
| T017: Rock or Bust | AC/DC | 2 → 1 | Reggae/progressive/Latin/classic/hard rock is only a partial fit to alternative rock; the artist exclusion is satisfied. |
| T025: 3WW | alt-J | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. Review caveat: Alternative Rock is supplied, but ambient/art-pop tags also occur; genre classification is metadata-based. |
| T139: Shame | Matchbox Twenty, Archilla, Matt | 2 → 2 | Alternative-rock evidence supports the requested style and no credited artist is Linkin Park. |
| T031: Create/Destroy | Art vs Science | 0 → 0 | The Linkin Park exclusion is satisfied, but electronic/mashup or indie-pop metadata does not substantiate alternative rock. |

## tobi-e02: Recommend hip hop without 21 Savage

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": ["21 Savage"], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T093: One Day | Jeru The Damaja | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T086: South | Hippo Campus | 0 → 0 | No 21 Savage credit appears, but alternative rock or pop does not meet the hip-hop request. |
| T078: Double Pump | Girl Talk | 2 → 1 | Hip-hop is one component of this electronic/rock mashup; the exclusion is satisfied but the musical fit is partial. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T022: 9-5ers Anthem | Aesop Rock | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T013: Like My Style | 50 Cent | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T205: Get Sum | Wiz Khalifa | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T089: The Message & the Money | Immortal Technique | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T151: Get Ur Freak On | Missy Elliott | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T036: Mad Izm | Channel Live | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T057: 3 a.m. | Eminem | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T141: Get Right Witcha | Migos | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. Review caveat: Migos has trap and hip-hop tags. The 21 Savage Numb seed has only Hip-Hop/Rap and artist tags; a specific shared trap style cannot be asserted from the seed record. |
| T131: Hott Bizness | Lyrics Born | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T015: Get A Hold | A Tribe Called Quest | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T080: Chloe | GROUPLOVE, Grouplove | 0 → 0 | No 21 Savage credit appears, but alternative rock or pop does not meet the hip-hop request. |
| T194: Darkest Things | The Submarines | 0 → 0 | No 21 Savage credit appears, but alternative rock or pop does not meet the hip-hop request. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T096: The Instrumental - feat. Jonah Matranga | Jonah Matranga, Lupe Fiasco | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T014: What Up Gangsta | 50 Cent | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T184: Whirlwind Pyramid | The D.O.C. | 2 → 2 | Hip-hop/rap is explicitly supported and no credited collaborator is the excluded artist 21 Savage. |
| T085: MMMBop | Hanson | 0 → 0 | No 21 Savage credit appears, but alternative rock or pop does not meet the hip-hop request. |
| T091: marshmallow | IU | 0 → 0 | No 21 Savage credit appears, but alternative rock or pop does not meet the hip-hop request. Review caveat: Upbeat is inferred from happy, bubblegum pop and teen-pop tags; listening confirmation remains needed. |

## tobi-e03: Recommend pop but exclude Adele

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": ["Adele"], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T193: Brightest Hour | The Submarines | 2 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. Review caveat: Skyfall similarity is an AI interpretation of piano/female-vocal/sad/calm/indie-pop tags, not an audio comparison or independently verified similarity fact. |
| T023: Get Lucky | Agapornis | 0 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. |
| T078: Double Pump | Girl Talk | 0 → 0 | The Adele exclusion is satisfied, but electronic, rap, rock, punk, or R&B-only metadata does not substantiate the requested pop genre. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T072: Mess | Get Scared | 0 → 0 | The Adele exclusion is satisfied, but electronic, rap, rock, punk, or R&B-only metadata does not substantiate the requested pop genre. |
| T157: Any Advice? | Nonpoint | 0 → 0 | The Adele exclusion is satisfied, but electronic, rap, rock, punk, or R&B-only metadata does not substantiate the requested pop genre. |
| T097: Get Like | Kehlani | 0 → 0 | The Adele exclusion is satisfied, but electronic, rap, rock, punk, or R&B-only metadata does not substantiate the requested pop genre. Review caveat: R&B/Soul is valid genre evidence but does not prove a mellow mood or similarity to Linkin Park, 21 Savage, or Adele. |
| T080: Chloe | GROUPLOVE, Grouplove | 0 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. |
| T206: Get Up | Xiu Xiu | 0 → 0 | The Adele exclusion is satisfied, but electronic, rap, rock, punk, or R&B-only metadata does not substantiate the requested pop genre. |
| T037: Get Mine, Get Yours | Christina Aguilera | 2 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. |
| T056: Uh Oh Hello | Elefant | 2 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 2 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 0 → 0 | The Adele exclusion is satisfied, but electronic, rap, rock, punk, or R&B-only metadata does not substantiate the requested pop genre. |
| T158: Television, Television | OK Go | 2 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. |
| T035: 3 | Britney Spears | 2 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. |
| T134: Ángel de amor | Maná | 0 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. |
| T165: Getaway | RuPaul | 0 → 0 | The Adele exclusion is satisfied, but electronic, rap, rock, punk, or R&B-only metadata does not substantiate the requested pop genre. |
| T091: marshmallow | IU | 2 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. Review caveat: Upbeat is inferred from happy, bubblegum pop and teen-pop tags; listening confirmation remains needed. |
| T139: Shame | Matchbox Twenty, Archilla, Matt | 2 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. |
| T191: What Happened to You? | The Offspring | 0 → 0 | The Adele exclusion is satisfied, but electronic, rap, rock, punk, or R&B-only metadata does not substantiate the requested pop genre. |
| T200: Morphine | Tony Moran, Michael Jackson | 2 → 2 | Pop, indie pop, or Latin Pop is supported and Adele is not a credited artist. No language or regional restriction was requested. |

## tobi-e04: Recommend metal without Metallica

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": ["Metallica"], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T070: Rent Money | Future | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. |
| T078: Double Pump | Girl Talk | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T133: Kings Of Metal, Kings of Metal | Manowar | 2 → 2 | Metal and its documented subgenres are supported; Metallica is not a credited artist even where mentioned in user tags. Review caveat: Metallica appears in a user tag, not the credited artist list; this is not an exclusion violation. |
| T177: Currency | The Black Angels | 0 → 1 | Hard rock or punk/post-hardcore is adjacent to metal; sparse metal tags amid stronger punk evidence do not justify an unqualified metal label. |
| T051: The Philosopher | Death | 2 → 2 | Metal and its documented subgenres are supported; Metallica is not a credited artist even where mentioned in user tags. |
| T080: Chloe | GROUPLOVE, Grouplove | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. |
| T171: Price To Play | Staind | 2 → 2 | Metal and its documented subgenres are supported; Metallica is not a credited artist even where mentioned in user tags. |
| T037: Get Mine, Get Yours | Christina Aguilera | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. |
| T056: Uh Oh Hello | Elefant | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 2 → 2 | Metal and its documented subgenres are supported; Metallica is not a credited artist even where mentioned in user tags. |
| T134: Ángel de amor | Maná | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. |
| T030: Despicable Heroes | Arch Enemy | 2 → 2 | Metal and its documented subgenres are supported; Metallica is not a credited artist even where mentioned in user tags. |
| T064: Autos, Moda Y Rock N Roll | Fandango | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. |
| T029: Metal On Metal | Anvil | 2 → 2 | Metal and its documented subgenres are supported; Metallica is not a credited artist even where mentioned in user tags. |
| T068: Suggestion | Fugazi | 0 → 1 | Hard rock or punk/post-hardcore is adjacent to metal; sparse metal tags amid stronger punk evidence do not justify an unqualified metal label. |
| T142: T-Shirt | Migos | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. |
| T032: Lies | Billy Talent | 0 → 1 | Hard rock or punk/post-hardcore is adjacent to metal; sparse metal tags amid stronger punk evidence do not justify an unqualified metal label. Review caveat: Energetic and alternative-rock evidence is clear, but the nu-metal/melodic-death-metal tags coexist with extensive punk tags and need listening confirmation. |
| T091: marshmallow | IU | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. Review caveat: Upbeat is inferred from happy, bubblegum pop and teen-pop tags; listening confirmation remains needed. |
| T105: Pedal to the Metal | Lazerhawk | 0 → 0 | The artist exclusion is satisfied, but no defensible metal match is established. A title containing Metal is not metal-genre evidence. |

## tobi-e05: Recommend rock music excluding Numb by Linkin Park

Context: `{"playlist": [], "likes": [], "dislikes": [], "excluded_artists": [], "excluded_tracks": [{"title": "Numb", "artist": "Linkin Park"}]}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T078: Double Pump | Girl Talk | 2 → 1 | Rock is a weaker crossover or incidental tag alongside dominant mashup, folk, jazz, hip-hop, or electronic styles; the specific song exclusion is satisfied. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T173: Serenade | Steve Miller Band | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. |
| T018: Rocker | AC/DC | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. |
| T183: Revolution Rock | The Clash | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. |
| T080: Chloe | GROUPLOVE, Grouplove | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. |
| T056: Uh Oh Hello | Elefant | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 0 → 0 | The excluded song is absent, but indie pop, Latin pop with a rock title, or electronic music lacks convincing rock evidence. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. |
| T134: Ángel de amor | Maná | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. |
| T064: Autos, Moda Y Rock N Roll | Fandango | 0 → 0 | The excluded song is absent, but indie pop, Latin pop with a rock title, or electronic music lacks convincing rock evidence. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | The excluded song is absent, but indie pop, Latin pop with a rock title, or electronic music lacks convincing rock evidence. |
| T185: Take It As It Comes | The Doors | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. Review caveat: Classic/psychedelic rock dominates despite one alternative-rock tag; partial alternative-rock or Linkin Park similarity is more defensible than full credit. |
| T071: Hate | Get Scared | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. |
| T104: Failure | Laura Marling | 2 → 1 | Rock is a weaker crossover or incidental tag alongside dominant mashup, folk, jazz, hip-hop, or electronic styles; the specific song exclusion is satisfied. Review caveat: Folk/acoustic/singer-songwriter evidence dominates despite folk-rock/pop/rock tags; contextual relevance is partial. |
| T181: Yoyo Waltz | The Cinematic Orchestra | 2 → 1 | Rock is a weaker crossover or incidental tag alongside dominant mashup, folk, jazz, hip-hop, or electronic styles; the specific song exclusion is satisfied. Review caveat: Jazz/downtempo dominates; the generic Rock tag warrants at most partial broad-rock relevance, not Numb-like relevance. |
| T154: Rock N Roll | Mos Def | 2 → 1 | Rock is a weaker crossover or incidental tag alongside dominant mashup, folk, jazz, hip-hop, or electronic styles; the specific song exclusion is satisfied. |
| T189: Rock Classics | The Knife | 2 → 1 | Rock is a weaker crossover or incidental tag alongside dominant mashup, folk, jazz, hip-hop, or electronic styles; the specific song exclusion is satisfied. Review caveat: Electronic/downtempo/synth-pop dominates despite Rock Classics title and alternative-rock tag. Broad rock is partial, specific Numb-like rock is unsupported. |
| T139: Shame | Matchbox Twenty, Archilla, Matt | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. |
| T163: Get Right | Pearl Jam | 2 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. |
| T055: rock + roll | EDEN | 0 → 2 | Rock/alternative/hard/progressive/Latin rock is verified, and the track is not the excluded Numb recording by Linkin Park. Review caveat: Alternative Rock is explicit. Sparse metadata is not grounds for treating a broad rock request as irrelevant, but cannot prove seed or mood similarity. |

## tobi-f01: Recommend rock; more like Numb by Linkin Park and less like Numb by 21 Savage

Context: `{"playlist": [], "likes": [{"title": "Numb", "artist": "Linkin Park"}], "dislikes": [{"title": "Numb", "artist": "21 Savage"}], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T078: Double Pump | Girl Talk | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T173: Serenade | Steve Miller Band | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. |
| T018: Rocker | AC/DC | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. |
| T183: Revolution Rock | The Clash | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. |
| T080: Chloe | GROUPLOVE, Grouplove | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. |
| T056: Uh Oh Hello | Elefant | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 0 → 0 | Dominant indie/Latin pop, electronic, or jazz metadata does not substantiate the requested Numb-like rock relationship; incidental rock tags are insufficient. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. |
| T134: Ángel de amor | Maná | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. |
| T064: Autos, Moda Y Rock N Roll | Fandango | 0 → 0 | Dominant indie/Latin pop, electronic, or jazz metadata does not substantiate the requested Numb-like rock relationship; incidental rock tags are insufficient. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | Dominant indie/Latin pop, electronic, or jazz metadata does not substantiate the requested Numb-like rock relationship; incidental rock tags are insufficient. |
| T185: Take It As It Comes | The Doors | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. Review caveat: Classic/psychedelic rock dominates despite one alternative-rock tag; partial alternative-rock or Linkin Park similarity is more defensible than full credit. |
| T071: Hate | Get Scared | 2 → 2 | Alternative metal plus alternative rock and post-hardcore gives specific metadata support for the liked Numb seed's style. |
| T104: Failure | Laura Marling | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. Review caveat: Folk/acoustic/singer-songwriter evidence dominates despite folk-rock/pop/rock tags; contextual relevance is partial. |
| T181: Yoyo Waltz | The Cinematic Orchestra | 2 → 0 | Dominant indie/Latin pop, electronic, or jazz metadata does not substantiate the requested Numb-like rock relationship; incidental rock tags are insufficient. Review caveat: Jazz/downtempo dominates; the generic Rock tag warrants at most partial broad-rock relevance, not Numb-like relevance. |
| T154: Rock N Roll | Mos Def | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. |
| T189: Rock Classics | The Knife | 2 → 0 | Dominant indie/Latin pop, electronic, or jazz metadata does not substantiate the requested Numb-like rock relationship; incidental rock tags are insufficient. Review caveat: Electronic/downtempo/synth-pop dominates despite Rock Classics title and alternative-rock tag. Broad rock is partial, specific Numb-like rock is unsupported. |
| T139: Shame | Matchbox Twenty, Archilla, Matt | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. |
| T163: Get Right | Pearl Jam | 2 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. |
| T055: rock + roll | EDEN | 0 → 1 | Broad rock or a crossover component offers partial relevance, but close similarity to the liked nu-metal/alternative-metal seed is not established. Less like 21 Savage does not ban all hip-hop. Review caveat: Alternative Rock is explicit. Sparse metadata is not grounds for treating a broad rock request as irrelevant, but cannot prove seed or mood similarity. |

## tobi-f02: Recommend hip hop; more like Numb by 21 Savage and less like Numb by Linkin Park

Context: `{"playlist": [], "likes": [{"title": "Numb", "artist": "21 Savage"}], "dislikes": [{"title": "Numb", "artist": "Linkin Park"}], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T192: Devil Music | The Pharcyde | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T100: JoJo's Chillin | Killer Mike | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T078: Double Pump | Girl Talk | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T098: DUCKWORTH. | Kendrick Lamar | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. Review caveat: Trap rap occurs with conscious/west-coast hip-hop tags, but the actual 21 Savage Numb seed has no trap tag. A shared seed style is not verified; feedback relevance is partial pending listening. |
| T013: Like My Style | 50 Cent | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T205: Get Sum | Wiz Khalifa | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T101: R.A.P. Music | Killer Mike | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T151: Get Ur Freak On | Missy Elliott | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T057: 3 a.m. | Eminem | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T141: Get Right Witcha | Migos | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. Review caveat: Migos has trap and hip-hop tags. The 21 Savage Numb seed has only Hip-Hop/Rap and artist tags; a specific shared trap style cannot be asserted from the seed record. |
| T063: President Heartbeat | Everything Everything | 0 → 0 | Alternative/pop/rock metadata lacks the requested hip-hop and positive-seed relationship. |
| T040: Pop's Rap | Common | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T015: Get A Hold | A Tribe Called Quest | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T012: Get In My Car | 50 Cent | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T102: 808 Man | KMD | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T083: Take A Look (At Yourself) | Guru's Jazzmatazz, Roy Ayers, Guru | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T014: What Up Gangsta | 50 Cent | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T088: Reverse Pimpology | Immortal Technique | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T204: November | Tyler, The Creator | 2 → 1 | Hip-hop/rap supplies broad relevance, but the actual liked Numb seed has only Hip-Hop/Rap and artist tags. Close track-level similarity, including a shared trap style, cannot be verified. The disliked Linkin Park track does not prohibit all genre crossover. |
| T085: MMMBop | Hanson | 0 → 0 | Alternative/pop/rock metadata lacks the requested hip-hop and positive-seed relationship. |

## tobi-f03: Recommend pop; I liked Skyfall by Adele

Context: `{"playlist": [], "likes": [{"title": "Skyfall - Full Length", "artist": "Adele"}], "dislikes": [], "excluded_artists": [], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T193: Brightest Hour | The Submarines | 2 → 2 | Female-vocal/piano, calm/sad pop with soundtrack associations gives a concrete metadata relationship to the liked Skyfall seed; confidence remains limited by noisy tags. Review caveat: Skyfall similarity is an AI interpretation of piano/female-vocal/sad/calm/indie-pop tags, not an audio comparison or independently verified similarity fact. |
| T078: Double Pump | Girl Talk | 0 → 0 | Electronic/mashup, punk, metal, or dominant guitar-rock metadata does not substantiate the requested pop-plus-Skyfall preference. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T072: Mess | Get Scared | 0 → 0 | Electronic/mashup, punk, metal, or dominant guitar-rock metadata does not substantiate the requested pop-plus-Skyfall preference. |
| T157: Any Advice? | Nonpoint | 0 → 0 | Electronic/mashup, punk, metal, or dominant guitar-rock metadata does not substantiate the requested pop-plus-Skyfall preference. |
| T080: Chloe | GROUPLOVE, Grouplove | 0 → 1 | Broad pop or adjacent soul/R&B supplies partial relevance, but the metadata does not establish a close cinematic soul/pop-ballad relationship to Skyfall. |
| T162: Options | PARTYNEXTDOOR | 0 → 1 | Broad pop or adjacent soul/R&B supplies partial relevance, but the metadata does not establish a close cinematic soul/pop-ballad relationship to Skyfall. Review caveat: R&B/Soul is meaningful but sparse. It is not evidence of a specific mood, hip-hop/trap style, or close seed similarity. |
| T037: Get Mine, Get Yours | Christina Aguilera | 2 → 1 | Broad pop or adjacent soul/R&B supplies partial relevance, but the metadata does not establish a close cinematic soul/pop-ballad relationship to Skyfall. |
| T056: Uh Oh Hello | Elefant | 2 → 1 | Broad pop or adjacent soul/R&B supplies partial relevance, but the metadata does not establish a close cinematic soul/pop-ballad relationship to Skyfall. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 2 → 1 | Broad pop or adjacent soul/R&B supplies partial relevance, but the metadata does not establish a close cinematic soul/pop-ballad relationship to Skyfall. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 0 → 0 | Electronic/mashup, punk, metal, or dominant guitar-rock metadata does not substantiate the requested pop-plus-Skyfall preference. |
| T134: Ángel de amor | Maná | 0 → 1 | Broad pop or adjacent soul/R&B supplies partial relevance, but the metadata does not establish a close cinematic soul/pop-ballad relationship to Skyfall. |
| T165: Getaway | RuPaul | 0 → 0 | Electronic/mashup, punk, metal, or dominant guitar-rock metadata does not substantiate the requested pop-plus-Skyfall preference. |
| T156: Showbiz | Muse | 0 → 0 | Electronic/mashup, punk, metal, or dominant guitar-rock metadata does not substantiate the requested pop-plus-Skyfall preference. Review caveat: Electronic is an isolated tag among dominant alternative/progressive-rock evidence; it does not justify strong Daft Punk similarity. |
| T073: Moving | Get Scared | 0 → 0 | Electronic/mashup, punk, metal, or dominant guitar-rock metadata does not substantiate the requested pop-plus-Skyfall preference. |
| T077: Whore | Get Scared | 0 → 0 | Electronic/mashup, punk, metal, or dominant guitar-rock metadata does not substantiate the requested pop-plus-Skyfall preference. |
| T104: Failure | Laura Marling | 2 → 1 | Broad pop or adjacent soul/R&B supplies partial relevance, but the metadata does not establish a close cinematic soul/pop-ballad relationship to Skyfall. Review caveat: Folk/acoustic/singer-songwriter evidence dominates despite folk-rock/pop/rock tags; contextual relevance is partial. |
| T091: marshmallow | IU | 2 → 1 | Broad pop or adjacent soul/R&B supplies partial relevance, but the metadata does not establish a close cinematic soul/pop-ballad relationship to Skyfall. Review caveat: Upbeat is inferred from happy, bubblegum pop and teen-pop tags; listening confirmation remains needed. |
| T139: Shame | Matchbox Twenty, Archilla, Matt | 2 → 1 | Broad pop or adjacent soul/R&B supplies partial relevance, but the metadata does not establish a close cinematic soul/pop-ballad relationship to Skyfall. |
| T191: What Happened to You? | The Offspring | 0 → 0 | Electronic/mashup, punk, metal, or dominant guitar-rock metadata does not substantiate the requested pop-plus-Skyfall preference. |
| T200: Morphine | Tony Moran, Michael Jackson | 2 → 1 | Broad pop or adjacent soul/R&B supplies partial relevance, but the metadata does not establish a close cinematic soul/pop-ballad relationship to Skyfall. |

## tobi-f04: Recommend alternative rock like In the End but no Metallica

Context: `{"playlist": [], "likes": [{"title": "In the End", "artist": "Linkin Park"}], "dislikes": [], "excluded_artists": ["Metallica"], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T078: Double Pump | Girl Talk | 2 → 0 | Electronic/hip-hop mashup, indie pop, or electronic metadata does not substantiate alternative rock like the liked seed, despite satisfying the exclusion. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T027: Isolation | Alter Bridge | 2 → 2 | Alternative-metal/nu-metal plus alternative-rock evidence closely matches the liked In the End seed, and Metallica is not credited. |
| T157: Any Advice? | Nonpoint | 2 → 2 | Alternative-metal/nu-metal plus alternative-rock evidence closely matches the liked In the End seed, and Metallica is not credited. |
| T183: Revolution Rock | The Clash | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. |
| T080: Chloe | GROUPLOVE, Grouplove | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. |
| T171: Price To Play | Staind | 2 → 2 | Alternative-metal/nu-metal plus alternative-rock evidence closely matches the liked In the End seed, and Metallica is not credited. |
| T056: Uh Oh Hello | Elefant | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 2 → 0 | Electronic/hip-hop mashup, indie pop, or electronic metadata does not substantiate alternative rock like the liked seed, despite satisfying the exclusion. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. |
| T158: Television, Television | OK Go | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. |
| T134: Ángel de amor | Maná | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. |
| T068: Suggestion | Fugazi | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. |
| T180: Music : Response | The Chemical Brothers | 0 → 0 | Electronic/hip-hop mashup, indie pop, or electronic metadata does not substantiate alternative rock like the liked seed, despite satisfying the exclusion. |
| T185: Take It As It Comes | The Doors | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. Review caveat: Classic/psychedelic rock dominates despite one alternative-rock tag; partial alternative-rock or Linkin Park similarity is more defensible than full credit. |
| T104: Failure | Laura Marling | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. Review caveat: Folk/acoustic/singer-songwriter evidence dominates despite folk-rock/pop/rock tags; contextual relevance is partial. |
| T154: Rock N Roll | Mos Def | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. |
| T139: Shame | Matchbox Twenty, Archilla, Matt | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. |
| T163: Get Right | Pearl Jam | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. |
| T031: Create/Destroy | Art vs Science | 0 → 0 | Electronic/hip-hop mashup, indie pop, or electronic metadata does not substantiate alternative rock like the liked seed, despite satisfying the exclusion. |
| T055: rock + roll | EDEN | 2 → 1 | Broad or adjacent rock is supported, but specific similarity to the liked rap-rock/nu-metal seed is weaker; the Metallica exclusion is satisfied. Review caveat: Alternative Rock is explicit. Sparse metadata is not grounds for treating a broad rock request as irrelevant, but cannot prove seed or mood similarity. |

## tobi-f05: Recommend energetic rock, more like Numb by Linkin Park, excluding 21 Savage

Context: `{"playlist": [], "likes": [{"title": "Numb", "artist": "Linkin Park"}], "dislikes": [], "excluded_artists": ["21 Savage"], "excluded_tracks": []}`

| Track | Artist | Draft → reviewed | Rationale |
| --- | --- | --- | --- |
| T107: Get Innocuous! | LCD Soundsystem | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T078: Double Pump | Girl Talk | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. Review caveat: Electronic/hip-hop mashup with a Rock tag; exact rock contribution and musical similarity require listening. An upbeat tag does not establish pop or R&B. |
| T157: Any Advice? | Nonpoint | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T183: Revolution Rock | The Clash | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T080: Chloe | GROUPLOVE, Grouplove | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T171: Price To Play | Staind | 2 → 2 | Explicit energetic plus alternative-rock/metal evidence supports both the requested mood and the liked Numb seed's rock style; 21 Savage is not credited. |
| T056: Uh Oh Hello | Elefant | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. Review caveat: Upbeat is inferred from happy, dance and pop/indie-pop tags rather than an explicit upbeat tag; review mood by listening. |
| T194: Darkest Things | The Submarines | 0 → 0 | Indie pop, predominantly mellow electronic, or hip-hop metadata does not substantiate energetic Numb-like rock. Review caveat: Indie/Pop/Alternative and mellow are documented, but no rock or R&B evidence is supplied; the original blanket alternative-rock labels were unsupported. |
| T054: Caught In A Web | Dream Theater | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T134: Ángel de amor | Maná | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T061: Halo | Evans Blue | 2 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. Review caveat: Energetic rock is explicit, but the record lacks specific nu-metal/alternative-metal evidence for the Numb feedback preference. |
| T021: Get The Lead Out | Aerosmith | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T154: Rock N Roll | Mos Def | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T032: Lies | Billy Talent | 2 → 2 | Explicit energetic plus alternative-rock/metal evidence supports both the requested mood and the liked Numb seed's rock style; 21 Savage is not credited. Review caveat: Energetic and alternative-rock evidence is clear, but the nu-metal/melodic-death-metal tags coexist with extensive punk tags and need listening confirmation. |
| T075: Suffer | Get Scared | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T189: Rock Classics | The Knife | 1 → 0 | Indie pop, predominantly mellow electronic, or hip-hop metadata does not substantiate energetic Numb-like rock. Review caveat: Electronic/downtempo/synth-pop dominates despite Rock Classics title and alternative-rock tag. Broad rock is partial, specific Numb-like rock is unsupported. |
| T139: Shame | Matchbox Twenty, Archilla, Matt | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T163: Get Right | Pearl Jam | 1 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. |
| T152: Get Dealt With | Mobb Deep | 0 → 0 | Indie pop, predominantly mellow electronic, or hip-hop metadata does not substantiate energetic Numb-like rock. |
| T055: rock + roll | EDEN | 0 → 1 | Rock/style or energy is only partly supported, or the specific relationship to Numb is weak. No artist exclusion is violated; aggressive/heavy alone is not verified energetic. Review caveat: Alternative Rock is explicit. Sparse metadata is not grounds for treating a broad rock request as irrelevant, but cannot prove seed or mood similarity. |
