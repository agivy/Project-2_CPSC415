from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# The two sentences
sentence1 = "A picture illustrating an article on a matter of public interest is not considered used for the purpose of trade or advertising within the prohibition of the statute (Gautier v. Pro-Football, 304 N. Y. 354, 359) unless it has no real relationship to the article "

sentence2 = "Accordingly, if a person brings a claim against a documentary filmmaker for the use of their image in the documentary, the claim would likely fail because a documentary is not deemed produced for the purposes of advertising or trade. See Gautier v. Pro-Football, Inc., 304 N.Y. 354, 359, 107 N.E.2d 485 (1952) (holding that a football player’s image was not used for trade or advertising when it appeared in a newsreel of a football game). Similarly, a claim brought against a journalist for the use of a person’s image in an article about a matter of public interest would also likely fail."

# Create embeddings
embedding1 = model.encode([sentence1])
embedding2 = model.encode([sentence2])

# Calculate cosine similarity
similarity = cosine_similarity(embedding1, embedding2)[0][0]

print(f"Cosine Similarity: {similarity:.4f}")