aae_rules = (
        "1. Null copula: Verbal copula is deleted (e.g., “He a delivery man” → “He's a delivery man”).\n"
        "2. Negative concord: Negatives agree with each other (e.g., “Nobody never say nothing” → “Nobody ever says anything”).\n"
        "3. Negative inversion: Auxiliary raises, but interpretation remains the same (e.g., “Don't nobody never say nothing to them” → “Nobody ever says anything to them”).\n"
        "4. Deletion of possessive /s/: Possessive markers are omitted, but the meaning remains (e.g., “His baby mama brother friend was there” → “His baby’s mother’s brother’s friend was there”).\n"
        "5. Habitual 'be like': describing something (e.g., “This song be like fire” → “This song is amazing”).\n"
        "6. Stressed 'been': short for have been doing something (e.g., “I been working out” → “I have been working out”).\n"
        "7. Preterite 'had': Signals the preterite or past action (e.g., “We had went to the store” → “We went to the store”).\n"
        "8. Question inversion in subordinate clauses: Inverts clauses similar to Standard English (e.g., “I was wondering did his white friend call?” → “I was wondering whether his white friend called”).\n"
        "9. Reduction of negation: Contraction of auxiliary verbs and negation (e.g., “I ain't even be feeling that” → “I don't much care for that”).\n"
        "10. Quotative 'talkin’ ’bout': Used as a verb of quotation (e.g., “She talkin’ ’bout he don’t live here no more” → “She's saying he doesn’t live here anymore”).\n"
        "11. Modal 'tryna': Short form of “trying to” (e.g., “I was tryna go to the store” → “I was planning on going to the store”).\n"
        "12. Expletive 'it': Used in place of “there” (e.g., “It’s a lot of money out there” → “There's a lot of money out there”).\n"
        "13. Shortening 'ing' words to 'in’': Any word that ends with 'ing' has to be converted into a in’ format (e.g. “he is playing” → “he is playin’”)."
    )

aave_prompt = (
    lambda src_text: f"You are an African American speaker fluent in both SAE (Standard American English) and AAVE (African American Vernacular English). Translate from SAE to AAVE in a way that feels natural and preserves the original meaning and tone. Avoid overly formal language, and aim for authenticity in the AAVE translation. Apply the following 13 translation rules:\n\n{aae_rules}\n\nYour output must also follow these guidelines:\n\n"
    "1. Only provide the translation. Do not mention or explain how the translation was done.\n"
    "2. Do not mention any of the 13 rules in your translation.\n"
    "3. Ensure the text sounds natural and realistic in AAVE.\n\n"
    f"Translate the following text: '{src_text}'"
    )

simple_prompt = (
    lambda src_text: f"You are a fluent English speaker. Rewrite a text so that it can be easily understood by a non-native speaker of English. Only provide the translation. Do not mention or explain how the translation was done. Translate the following text: '{src_text}'"
)

# This is the complex prompt. It's currently not used as it had worse SARI / BERT score results
complex_prompt = (
    lambda src_text: f"You are a fluent English speaker and a professional simplified text writer. Your task is to simplify the language and structure of the given text content to make it more accesibe to pupils, non-native speakers, or people with cognitive impairments. Restate the original text in simpler and easier-to-understand language without changing its meaning as much as possible. You can change paragraph or sentence structure, and replace complex and uncommon expressions with simple and common ones. Only provide the translation. Do not mention or explain how the translation was done. Simplify the following text: '{src_text}'"
)

vocabulary = "come, get, give, go, keep, let, make, put, seem, take, be, do, have, say, see, send, may, will, about, across, after, against, among, at, before, between, by, down, from, in, off, on, over, through, to, under, up, with, as, for, of, till, than, a, the, all, any, every, no, other, some, such, that, this, I, he, you, who, and, because, but, or, if, though, while, how, when, where, why, again, ever, far, forward, here, near, now, out, still, then, there, together, well, almost, enough, even, little, much, not, only, quite, so, very, tomorrow, yesterday, north, south, east, west, please, yes, account, act, addition, adjustment, advertisement, agreement, air, amount, amusement, animal, answer, apparatus, approval, argument, art, attack, attempt, attention, attraction, authority, back, balance, base, behaviour, belief, birth, bit, bite, blood, blow, body, brass, bread, breath, brother, building, burn, burst, business, butter, canvas, care, cause, chalk, chance, change, cloth, coal, colour, comfort, committee, company, comparison, competition, condition, connection, control, cook, copper, copy, cork, cotton, cough, country, cover, crack, credit, crime, crush, cry, current, curve, damage, danger, daughter, day, death, debt, decision, degree, design, desire, destruction, detail, development, digestion, direction, discovery, discussion, disease, disgust, distance, distribution, division, doubt, drink, driving, dust, earth, edge, education, effect, end, error, event, example, exchange, existence, expansion, experience, expert, fact, fall, family, father, fear, feeling, fiction, field, fight, fire, flame, flight, flower, fold, food, force, form, friend, front, fruit, glass, gold, government, grain, grass, grip, group, growth, guide, harbour, harmony, hate, hearing, heat, help, history, hole, hope, hour, humour, ice, idea, impulse, increase, industry, ink, insect, instrument, insurance, interest, invention, iron, jelly, join, journey, judge, jump, kick, kiss, knowledge, land, language, laugh, law, lead, learning, leather, letter, level, lift, light, limit, linen, liquid, list, look, loss, love, machine, man, manager, mark, market, mass, meal, measure, meat, meeting, memory, metal, middle, milk, mind, mine, minute, mist, money, month, morning, mother, motion, mountain, move, music, name, nation, need, news, night, noise, note, number, observation, offer, oil, operation, opinion, order, organization, ornament, owner, page, pain, paint, paper, part, paste, payment, peace, person, place, plant, play, pleasure, point, poison, polish, porter, position, powder, power, price, print, process, produce, profit, property, prose, protest, pull, punishment, purpose, push, quality, question, rain, range, rate, ray, reaction, reading, reason, record, regret, relation, religion, representative, request, respect, rest, reward, rhythm, rice, river, road, roll, room, rub, rule, run, salt, sand, scale, science, sea, seat, secretary, selection, self, sense, servant, sex, shade, shake, shame, shock, side, sign, silk, silver, sister, size, sky, sleep, slip, slope, smash, smell, smile, smoke, sneeze, snow, soap, society, son, song, sort, sound, soup, space, stage, start, statement, steam, steel, step, stitch, stone, stop, story, stretch, structure, substance, sugar, suggestion, summer, support, surprise, swim, system, talk, taste, tax, teaching, tendency, test, theory, thing, thought, thunder, time, tin, top, touch, trade, transport, trick, trouble, turn, twist, unit, use, value, verse, vessel, view, voice, walk, war, wash, waste, water, wave, wax, way, weather, week, weight, wind, wine, winter, woman, wood, wool, word, work, wound, writing, year, angle, ant, apple, arch, arm, army, baby, bag, ball, band, basin, basket, bath, bed, bee, bell, berry, bird, blade, board, boat, bone, book, boot, bottle, box, boy, brain, brake, branch, brick, bridge, brush, bucket, bulb, button, cake, camera, card, cart, carriage, cat, chain, cheese, chest, chin, church, circle, clock, cloud, coat, collar, comb, cord, cow, cup, curtain, cushion, dog, door, drain, drawer, dress, drop, ear, egg, engine, eye, face, farm, feather, finger, fish, flag, floor, fly, foot, fork, fowl, frame, garden, girl, glove, goat, gun, hair, hammer, hand, hat, head, heart, hook, horn, horse, hospital, house, island, jewel, kettle, key, knee, knife, knot, leaf, leg, library, line, lip, lock, map, match, monkey, moon, mouth, muscle, nail, neck, needle, nerve, net, nose, nut, office, orange, oven, parcel, pen, pencil, picture, pig, pin, pipe, plane, plate, plough, pocket, pot, potato, prison, pump, rail, rat, receipt, ring, rod, roof, root, sail, school, scissors, screw, seed, sheep, shelf, ship, shirt, shoe, skin, skirt, snake, sock, spade, sponge, spoon, spring, square, stamp, star, station, stem, stick, stocking, stomach, store, street, sun, table, tail, thread, throat, thumb, ticket, toe, tongue, tooth, town, train, tray, tree, trousers, umbrella, wall, watch, wheel, whip, whistle, window, wing, wire, worm, able, acid, angry, automatic, beautiful, black, boiling, bright, broken, brown, cheap, chemical, chief, clean, clear, common, complex, conscious, cut, deep, dependent, early, elastic, electric, equal, fat, fertile, first, fixed, flat, free, frequent, full, general, good, great, grey, hanging, happy, hard, healthy, high, hollow, important, kind, like, living, long, male, married, material, medical, military, natural, necessary, new, normal, open, parallel, past, physical, political, poor, possible, present, private, probable, quick, quiet, ready, red, regular, responsible, right, round, same, second, separate, serious, sharp, smooth, sticky, stiff, straight, strong, sudden, sweet, tall, thick, tight, tired, true, violent, waiting, warm, wet, wide, wise, yellow, young, awake, bad, bent, bitter, blue, certain, cold, complete, cruel, dark, dead, dear, delicate, different, dirty, dry, false, feeble, female, foolish, future, green, ill, last, late, left, loose, loud, low, mixed, narrow, old, opposite, public, rough, sad, safe, secret, short, shut, simple, slow, small, soft, solid, special, strange, thin, white, wrong"

basic_rules = f"""
1. Plurals are formed with a trailing \"S\". The normal exceptions of standard English also apply, notably \"ES\" and \"IES\".
2. There are four derivatives for the 300 nouns: -\"ER\" and -\"ING\", and two adjectives, -\"ING\" and -\"ED\".
3. Adverbs use -\"LY\" from qualifiers.
4. Degree is expressed with \"MORE\" and \"MOST\". Be prepared to find -\"ER\" and -\"EST\" in common usage.
5. Negative adjectives are formed with \"UN\"-.
6. Questions are formed by inversion and by \"DO\".
7. Operators and pronouns conjugate in full.
8. Compound words may be combined from two nouns (milkman) or a noun and a directive (sundown).
9. Measurement, numerals, currency, calendar, and international terms are in English form.
10. Technical expressions required and customary for the immediate task are included in the locally used form."""

# This is the Basic English prompt. It's currently not used as it had worse SARI / BERT score results
basic_english_prompt = lambda src_text: (
    f"You are a fluent English speaker. Translate from Standard English to Basis English in a way that feels natural and preserves the original meaning and tone. You should use the following 850-word vocabulary of Basic English:\n\n{vocabulary}\n\nUse these 10 rules of grammar for Basic English:\n{basic_rules}\n\nYour output must also follow these guidelines::\n\n"
    "1. Only provide the translation. Do not mention or explain how the translation was done.\n"
    "2. Do not mention any of the 10 rules in your translation.\n"
    "3. Ensure the text sounds natural and realistic in Basic English.\n\n"
    f"Translate the following text: '{src_text}'"
)


error_prompt = lambda question, answer: (
    "You are a fact checker.\nYou will be given a question-answer pair.\nYour task is to rewrite the answer while inserting exactly ONE subtle mistake. That mistake should make the answer worse.\nOnly provide the new answer. Do not mention or explain how the error was introduced. You should not modify any content apart from the error.\n\n"
    f"Your question is: '{question}'\n"
    f"Rewrite and introduce a mistake for the following answer: '{answer}'"
)

extract_reasons_prompt = lambda judge_reasoning, winner, loser: (f"""Analyze this judge's reasoning and extract specific strengths and weaknesses mentioned.

        Judge's reasoning:
        '{judge_reasoning}'

        Winner: '{winner}'
        Loser: '{loser}'

        Extract and list:
        1. Specific strengths mentioned about the WINNER (be concise, one short phrase per strength)
        2. Specific weaknesses mentioned about the LOSER (be concise, one short phrase per weakness)
        
        Do NOT include information that could lead to identifying the question. The strengths and weaknesses should be general and applicable to various contexts.
        
        Good example output:
        
        {{
        "strengths": ["comprehensive and detailed explanation", "provides additional context", ...],
        "weaknesses": ["less comprehensive", "lacks detail", "does not delve as deeply"...]
        }}
        
        Bad example output:
        
        {{
        "strengths": ["provides insights on accessibility of safety", "mentions importance of variety and balance", ...],
        "weaknesses": ["doesn't mention loose wires and broken solder joint", "fails to address user safety directly", "does not delve as deeply into historians' interpretations"...]
        }}

        Format your response as JSON:
        {{
        "strengths": ["strength 1", "strength 2", ...],
        "weaknesses": ["weakness 1", "weakness 2", ...]
        }}

        Only include explicitly mentioned qualities. Be specific and concise.""")

cluster_descriptions_prompt = """
        Here is a cluster of descriptions of a model answer: \n[DOCUMENTS]
        The cluster is described by the following keywords: [KEYWORDS]

        Based on the above information, can you give one short concise description (1-3 words) that summarizes the main idea of this cluster.
        It should also make clear whether the description is a strength or a weakness of the model answer, but do not explicitly mention it, i.e. the words "strength" or "weakness" should not be in the description directly.
        Provide only the concise description without any additional explanation.
"""