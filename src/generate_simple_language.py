import json

import yaml

from calculate_readability_metrics import load_onestopqa
from model import get_model
from utils import read_file, write_file

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
    my_api_key = config["openai_key"]

# vocabulary = "come, get, give, go, keep, let, make, put, seem, take, be, do, have, say, see, send, may, will, about, across, after, against, among, at, before, between, by, down, from, in, off, on, over, through, to, under, up, with, as, for, of, till, than, a, the, all, any, every, no, other, some, such, that, this, I, he, you, who, and, because, but, or, if, though, while, how, when, where, why, again, ever, far, forward, here, near, now, out, still, then, there, together, well, almost, enough, even, little, much, not, only, quite, so, very, tomorrow, yesterday, north, south, east, west, please, yes, account, act, addition, adjustment, advertisement, agreement, air, amount, amusement, animal, answer, apparatus, approval, argument, art, attack, attempt, attention, attraction, authority, back, balance, base, behaviour, belief, birth, bit, bite, blood, blow, body, brass, bread, breath, brother, building, burn, burst, business, butter, canvas, care, cause, chalk, chance, change, cloth, coal, colour, comfort, committee, company, comparison, competition, condition, connection, control, cook, copper, copy, cork, cotton, cough, country, cover, crack, credit, crime, crush, cry, current, curve, damage, danger, daughter, day, death, debt, decision, degree, design, desire, destruction, detail, development, digestion, direction, discovery, discussion, disease, disgust, distance, distribution, division, doubt, drink, driving, dust, earth, edge, education, effect, end, error, event, example, exchange, existence, expansion, experience, expert, fact, fall, family, father, fear, feeling, fiction, field, fight, fire, flame, flight, flower, fold, food, force, form, friend, front, fruit, glass, gold, government, grain, grass, grip, group, growth, guide, harbour, harmony, hate, hearing, heat, help, history, hole, hope, hour, humour, ice, idea, impulse, increase, industry, ink, insect, instrument, insurance, interest, invention, iron, jelly, join, journey, judge, jump, kick, kiss, knowledge, land, language, laugh, law, lead, learning, leather, letter, level, lift, light, limit, linen, liquid, list, look, loss, love, machine, man, manager, mark, market, mass, meal, measure, meat, meeting, memory, metal, middle, milk, mind, mine, minute, mist, money, month, morning, mother, motion, mountain, move, music, name, nation, need, news, night, noise, note, number, observation, offer, oil, operation, opinion, order, organization, ornament, owner, page, pain, paint, paper, part, paste, payment, peace, person, place, plant, play, pleasure, point, poison, polish, porter, position, powder, power, price, print, process, produce, profit, property, prose, protest, pull, punishment, purpose, push, quality, question, rain, range, rate, ray, reaction, reading, reason, record, regret, relation, religion, representative, request, respect, rest, reward, rhythm, rice, river, road, roll, room, rub, rule, run, salt, sand, scale, science, sea, seat, secretary, selection, self, sense, servant, sex, shade, shake, shame, shock, side, sign, silk, silver, sister, size, sky, sleep, slip, slope, smash, smell, smile, smoke, sneeze, snow, soap, society, son, song, sort, sound, soup, space, stage, start, statement, steam, steel, step, stitch, stone, stop, story, stretch, structure, substance, sugar, suggestion, summer, support, surprise, swim, system, talk, taste, tax, teaching, tendency, test, theory, thing, thought, thunder, time, tin, top, touch, trade, transport, trick, trouble, turn, twist, unit, use, value, verse, vessel, view, voice, walk, war, wash, waste, water, wave, wax, way, weather, week, weight, wind, wine, winter, woman, wood, wool, word, work, wound, writing, year, angle, ant, apple, arch, arm, army, baby, bag, ball, band, basin, basket, bath, bed, bee, bell, berry, bird, blade, board, boat, bone, book, boot, bottle, box, boy, brain, brake, branch, brick, bridge, brush, bucket, bulb, button, cake, camera, card, cart, carriage, cat, chain, cheese, chest, chin, church, circle, clock, cloud, coat, collar, comb, cord, cow, cup, curtain, cushion, dog, door, drain, drawer, dress, drop, ear, egg, engine, eye, face, farm, feather, finger, fish, flag, floor, fly, foot, fork, fowl, frame, garden, girl, glove, goat, gun, hair, hammer, hand, hat, head, heart, hook, horn, horse, hospital, house, island, jewel, kettle, key, knee, knife, knot, leaf, leg, library, line, lip, lock, map, match, monkey, moon, mouth, muscle, nail, neck, needle, nerve, net, nose, nut, office, orange, oven, parcel, pen, pencil, picture, pig, pin, pipe, plane, plate, plough, pocket, pot, potato, prison, pump, rail, rat, receipt, ring, rod, roof, root, sail, school, scissors, screw, seed, sheep, shelf, ship, shirt, shoe, skin, skirt, snake, sock, spade, sponge, spoon, spring, square, stamp, star, station, stem, stick, stocking, stomach, store, street, sun, table, tail, thread, throat, thumb, ticket, toe, tongue, tooth, town, train, tray, tree, trousers, umbrella, wall, watch, wheel, whip, whistle, window, wing, wire, worm, able, acid, angry, automatic, beautiful, black, boiling, bright, broken, brown, cheap, chemical, chief, clean, clear, common, complex, conscious, cut, deep, dependent, early, elastic, electric, equal, fat, fertile, first, fixed, flat, free, frequent, full, general, good, great, grey, hanging, happy, hard, healthy, high, hollow, important, kind, like, living, long, male, married, material, medical, military, natural, necessary, new, normal, open, parallel, past, physical, political, poor, possible, present, private, probable, quick, quiet, ready, red, regular, responsible, right, round, same, second, separate, serious, sharp, smooth, sticky, stiff, straight, strong, sudden, sweet, tall, thick, tight, tired, true, violent, waiting, warm, wet, wide, wise, yellow, young, awake, bad, bent, bitter, blue, certain, cold, complete, cruel, dark, dead, dear, delicate, different, dirty, dry, false, feeble, female, foolish, future, green, ill, last, late, left, loose, loud, low, mixed, narrow, old, opposite, public, rough, sad, safe, secret, short, shut, simple, slow, small, soft, solid, special, strange, thin, white, wrong"

# rules = (f"""
# 1. Plurals are formed with a trailing \"S\". The normal exceptions of standard English also apply, notably \"ES\" and \"IES\".
# 2. There are four derivatives for the 300 nouns: -\"ER\" and -\"ING\", and two adjectives, -\"ING\" and -\"ED\".
# 3. Adverbs use -\"LY\" from qualifiers.
# 4. Degree is expressed with \"MORE\" and \"MOST\". Be prepared to find -\"ER\" and -\"EST\" in common usage.
# 5. Negative adjectives are formed with \"UN\"-.
# 6. Questions are formed by inversion and by \"DO\".
# 7. Operators and pronouns conjugate in full.
# 8. Compound words may be combined from two nouns (milkman) or a noun and a directive (sundown).
# 9. Measurement, numerals, currency, calendar, and international terms are in English form.
# 10. Technical expressions required and customary for the immediate task are included in the locally used form.""")

# This is the Basic English prompt. It's currently not used as it had worse SARI / BERT score results
# user_input_basic_english = lambda src_text:(
#     f"You are a fluent English speaker. Translate from Standard English to Basis English in a way that feels natural and preserves the original meaning and tone. You should use the following 850-word vocabulary of Basic English:\n\n{vocabulary}\n\nUse these 10 rules of grammar for Basic English:\n{rules}\n\nYour output must also follow these guidelines::\n\n"
#     "1. Only provide the translation. Do not mention or explain how the translation was done.\n"
#     "2. Do not mention any of the 10 rules in your translation.\n"
#     "3. Ensure the text sounds natural and realistic in Basic English.\n\n"
#     f"Translate the following text: '{src_text}'")


user_input = (
    lambda src_text: f"You are a fluent English speaker. Rewrite a text so that it can be easily understood by a non-native speaker of English. Only provide the translation. Do not mention or explain how the translation was done. Translate the following text: '{src_text}'"
)

# This is the complex prompt. It's currently not used as it had worse SARI / BERT score results
# user_input_complex = lambda src_text: f"You are a fluent English speaker and a professional simplified text writer. Your task is to simplify the language and structure of the given text content to make it more accesibe to pupils, non-native speakers, or people with cognitive impairments. Restate the original text in simpler and easier-to-understand language without changing its meaning as much as possible. You can change paragraph or sentence structure, and replace complex and uncommon expressions with simple and common ones. Only provide the translation. Do not mention or explain how the translation was done. Simplify the following text: '{src_text}'"

model_name_or_path = "gpt-4.1"

original_file = "data/generated_answers/gpt-4.1-answers.json"
original_dicts = read_file(original_file)

new_dicts = [dict(dictionary) for dictionary in original_dicts]

texts = [
    [entry["question"] for entry in original_dicts],
    [entry["answers"]["answer1"]["answer"] for entry in original_dicts],
    [entry["answers"]["answer2"]["answer"] for entry in original_dicts],
]
prompts = [
    [user_input(question) for question in texts[0]],
    [user_input(answer) for answer in texts[1]],
    [user_input(answer) for answer in texts[2]],
]

model = get_model(
    model_name_or_path=model_name_or_path,
    config=config,
)

system_prompts = [""] * len(prompts[0])

responses = [
    model.generate(system_prompts=system_prompts, input_texts=prompts[0]),
    model.generate(system_prompts=system_prompts, input_texts=prompts[1]),
    model.generate(system_prompts=system_prompts, input_texts=prompts[2]),
]

for i, dictionary in enumerate(new_dicts):
    dictionary["model_name"] = model_name_or_path
    dictionary["question"] = responses[0][i][0]
    dictionary["answers"]["answer1"]["answer"] = responses[1][i][0]
    dictionary["answers"]["answer2"]["answer"] = responses[2][i][0]

write_file(original_file.replace(".json", "_basic.json"), new_dicts)

### The following code is for generating simple language from the OneStopQA data
# articles = load_onestopqa(reference=False)
# articles_truncated = [
#     " ".join(paragraph[:3]) for paragraph in articles]
# prompts = [user_input(article) for article in articles_truncated]
# system_prompts = [""] * len(prompts)
# responses = model.generate(
#     system_prompts=system_prompts,
#     input_texts=prompts,
# )
# simple_language_paragraphs = [response[0] for response in responses]
# # write output list to file
# with open("data/readability_metrics/onestopqa_test_data/complex_prompt.json", "w") as file:
#     json.dump(simple_language_paragraphs, file, indent=4)
