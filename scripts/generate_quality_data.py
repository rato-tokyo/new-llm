# -*- coding: utf-8 -*-
"""Generate high-quality diverse dataset for New-LLM training

Goals:
- 500+ sentences
- Diverse sentence lengths (5-30 words)
- Complex sentence structures (subordinate clauses, conjunctions)
- Rich vocabulary (300+ unique words)
- Various tenses and grammatical patterns
"""

import random

# Building blocks for diverse sentences
subjects = [
    "the scientist", "a young student", "the experienced teacher", "my best friend",
    "the talented musician", "an ambitious entrepreneur", "the dedicated doctor",
    "a curious child", "the skilled programmer", "an elderly professor",
    "the creative artist", "a brave firefighter", "the patient nurse",
    "an innovative engineer", "the wise philosopher", "a determined athlete",
    "the passionate writer", "an honest politician", "the caring parent",
    "a brilliant researcher", "the humble farmer", "an adventurous explorer",
    "the thoughtful librarian", "a hardworking chef", "the famous actor",
    "an inspired poet", "the reliable mechanic", "a talented designer",
    "the successful businessman", "an enthusiastic volunteer"
]

verbs_present = [
    "works", "studies", "teaches", "learns", "creates", "builds", "discovers",
    "explores", "develops", "writes", "reads", "thinks", "believes", "hopes",
    "tries", "practices", "performs", "manages", "leads", "follows", "helps",
    "supports", "encourages", "inspires", "motivates", "challenges", "questions",
    "investigates", "analyzes", "designs", "plans", "organizes", "coordinates"
]

verbs_past = [
    "worked", "studied", "taught", "learned", "created", "built", "discovered",
    "explored", "developed", "wrote", "read", "thought", "believed", "hoped",
    "tried", "practiced", "performed", "managed", "led", "followed", "helped",
    "supported", "encouraged", "inspired", "motivated", "challenged", "questioned",
    "investigated", "analyzed", "designed", "planned", "organized", "coordinated"
]

verbs_future = [
    "will work", "will study", "will teach", "will learn", "will create",
    "will build", "will discover", "will explore", "will develop", "will write"
]

objects = [
    "complex problems", "innovative solutions", "important discoveries",
    "beautiful artwork", "useful tools", "new technologies", "better methods",
    "valuable lessons", "interesting theories", "practical applications",
    "crucial experiments", "detailed reports", "creative projects",
    "challenging questions", "meaningful connections", "significant improvements",
    "effective strategies", "clear explanations", "helpful resources",
    "inspiring stories", "accurate predictions", "reliable systems"
]

locations = [
    "at the university", "in the laboratory", "at home", "in the library",
    "at the office", "in the classroom", "at the hospital", "in the studio",
    "at the conference", "in the workshop", "at the museum", "in the field",
    "at the factory", "in the garden", "at the beach", "in the mountains"
]

time_expressions = [
    "every day", "last week", "next month", "this morning", "yesterday",
    "tomorrow", "last year", "in the future", "recently", "nowadays",
    "currently", "previously", "soon", "eventually", "frequently",
    "occasionally", "rarely", "sometimes", "often", "always"
]

conjunctions = ["because", "although", "while", "when", "if", "unless", "since", "after", "before"]

subordinate_clauses = [
    "it helps people understand the world better",
    "it contributes to scientific progress",
    "it makes life easier for everyone",
    "it solves real-world problems",
    "it advances human knowledge",
    "it improves quality of life",
    "it creates new opportunities",
    "it benefits society as a whole",
    "it addresses important challenges",
    "it promotes innovation and growth"
]

def generate_simple_sentence():
    """Generate a simple sentence (5-10 words)"""
    subj = random.choice(subjects)
    verb = random.choice(verbs_present)
    obj = random.choice(objects)
    return f"{subj} {verb} {obj}"

def generate_with_time():
    """Generate sentence with time expression (8-12 words)"""
    subj = random.choice(subjects)
    verb = random.choice(verbs_past + verbs_present)
    obj = random.choice(objects)
    time = random.choice(time_expressions)
    return f"{subj} {verb} {obj} {time}"

def generate_with_location():
    """Generate sentence with location (10-15 words)"""
    subj = random.choice(subjects)
    verb = random.choice(verbs_present + verbs_past)
    obj = random.choice(objects)
    loc = random.choice(locations)
    return f"{subj} {verb} {obj} {loc}"

def generate_complex_sentence():
    """Generate complex sentence with subordinate clause (15-25 words)"""
    subj = random.choice(subjects)
    verb = random.choice(verbs_present)
    obj = random.choice(objects)
    conj = random.choice(conjunctions)
    clause = random.choice(subordinate_clauses)
    return f"{subj} {verb} {obj} {conj} {clause}"

def generate_compound_sentence():
    """Generate compound sentence (12-20 words)"""
    sent1 = generate_simple_sentence()
    sent2 = generate_simple_sentence()
    connector = random.choice(["and", "but", "so", "yet", "or"])
    return f"{sent1} {connector} {sent2}"

def generate_with_all_elements():
    """Generate rich sentence with multiple elements (18-30 words)"""
    subj = random.choice(subjects)
    verb = random.choice(verbs_past)
    obj = random.choice(objects)
    loc = random.choice(locations)
    time = random.choice(time_expressions)
    conj = random.choice(conjunctions)
    clause = random.choice(subordinate_clauses)
    return f"{subj} {verb} {obj} {loc} {time} {conj} {clause}"

# Generate dataset
sentences = []

# Distribution of sentence types (total 500)
for _ in range(80):
    sentences.append(generate_simple_sentence())

for _ in range(100):
    sentences.append(generate_with_time())

for _ in range(100):
    sentences.append(generate_with_location())

for _ in range(120):
    sentences.append(generate_complex_sentence())

for _ in range(60):
    sentences.append(generate_compound_sentence())

for _ in range(40):
    sentences.append(generate_with_all_elements())

# Shuffle to mix sentence types
random.shuffle(sentences)

# Write to file
output_path = "data/sample_texts.txt"
with open(output_path, 'w') as f:
    for sentence in sentences:
        f.write(sentence + '\n')

print(f"Generated {len(sentences)} sentences")
print(f"Saved to {output_path}")

# Calculate statistics
total_words = sum(len(s.split()) for s in sentences)
unique_words = len(set(' '.join(sentences).lower().split()))
avg_length = total_words / len(sentences)

print(f"\nDataset Statistics:")
print(f"  Total sentences: {len(sentences)}")
print(f"  Total words: {total_words}")
print(f"  Unique words: {unique_words}")
print(f"  Average sentence length: {avg_length:.1f} words")
print(f"  Min length: {min(len(s.split()) for s in sentences)} words")
print(f"  Max length: {max(len(s.split()) for s in sentences)} words")
