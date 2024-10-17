from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-large')
print('loaded tokenizer')
model = T5ForConditionalGeneration.from_pretrained('t5-large')
print('loaded model')
def answer_question(question, context):
    input_text = f"question: {question}  context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(
        inputs.input_ids,
        max_length=712,  # Increase this value to allow longer responses
        min_length=100,  # Set a minimum length for detailed answers
        num_beams=5,  # Use beam search to improve quality
        early_stopping=True,  # Stop generating if the model finishes the answer
        no_repeat_ngram_size=3,  # Prevent repeating phrases,
        do_sample=True,
        temperature=0.7,  # Adjust temperature for focused output
        top_p=0.9,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

context="Potential impacts of synthetic food dyes on activity and attention in children: a review of the human and animal evidence." \
         "Authors: Mark D. Miller, Craig Steinmaus, Mari S. Golub, Rosemary Castorina, Ruwan Thilakartne, Asa Bradman, and Melanie A. Marty." \
         "Abstract: Concern that synthetic food dyes may impact behavior in children prompted a review by the California Office of Environmental Health Hazard Assessment (OEHHA)."\
"OEHHA conducted a systematic review of the epidemiologic research on synthetic food dyes and neurobehavioral outcomes in children with or without identified behavioral disorders (particularly attention and activity)."\
"We also conducted a search of the animal toxicology literature to identify studies of neurobehavioral effects in laboratory animals exposed to synthetic food dyes."\
"Finally, we conducted a hazard characterization of the potential neurobehavioral impacts of food dye consumption."\
"We identified 27 clinical trials of children exposed to synthetic food dyes in this review, of which 25 were challenge studies."\
"All studies used a cross-over design and most were double blinded and the cross-over design was randomized."\
"Sixteen (64%) out of 25 challenge studies identified some evidence of a positive association, and in 13 (52%) the association was statistically significant."\
"These studies support a relationship between food dye exposure and adverse behavioral outcomes in children."\
"Animal toxicology literature provides additional support for effects on behavior."\
"Together, the human clinical trials and animal toxicology literature support an association between synthetic food dyes and behavioral impacts in children."\
"The current Food and Drug Administration (FDA) acceptable daily intakes are based on older studies that were not designed to assess the types of behavioral effects observed in children."\
"For four dyes where adequate dose-response data from animal and human studies were available, comparisons of the effective doses in studies that measured behavioral or brain effects following exposure to synthetic food dyes indicate that the basis of the ADIs may not be adequate to protect neurobehavior in susceptible children."\
"There is a need to re-evaluate exposure in children and for additional research to provide a more complete database for establishing ADIs protective of neurobehavioral effects."\
"Keywords: Synthetic food dyes, Children, Behavior, Clinical trials, Animal toxicology"\
"Background: Concerns about possible associations between exposure to synthetic food dyes and the exacerbation of symptoms of Attention Deficit/Hyperactivity Disorder (ADHD) in children have surfaced periodically since the 1970s."\
"The concern prompted the California legislature to request a review by the California Environmental Protection Agency’s Office of Environmental Health Hazard Assessment (OEHHA) of available studies to evaluate whether the synthetic food dyes currently allowed in foods and medications in the United States impact neurobehavior in children [1]."\
"This paper provides an overview of key portions of OEHHA’s peer-reviewed assessment, specifically the evaluation of the clinical trials of synthetic food dyes in children and available animal toxicology studies, as well as discussion of our hazard characterization and the possible public health implications of our findings."\
"Our evaluation focused on seven of the nine food dyes subject to FD&C batch certification by the US Food and Drug Administration (FDA) and approved for general use in food in the US (Table 1)."\
"These seven dyes contribute nearly all of the exposure to synthetic food dyes for the general US public [1]."\
"The term “FD&C batch-certified” refers to the Food Drug and Cosmetic Act requirements for chemical analysis of each manufactured batch of food dye to ensure that specific contaminants are present below legal limits."\
"OEHHA evaluated the literature to determine whether there is any evidence supporting the association of exposure to synthetic food dyes with adverse neurobehavioral impacts in children in the general population with or without a diagnosis of ADHD."\
"Methods: The literature review methods were designed to identify all the literature most relevant to the assessment of evidence on the neurological or neurobehavioral effects of the synthetic food dyes listed in Table 1."\
"The search was executed to identify peer-reviewed open-source and proprietary journal articles, print and digital books, reports, and gray literature that potentially reported relevant toxicological and epidemiological information."\
"We also included Citrus Red No. 2 and Orange B/CI Acid Orange in the search terms since these food dyes are part of anoverlapping literature that might contain information on the commonly used FD&C synthetic food dyes."\
"PubMed MeSH browser (PubMed MeSH browser) and PubChem were used to identify subject headings, other index terms and synonyms for the food dyes of interest and their metabolites, as well as for the concepts related to exposure, food, mechanisms of action, and neurological outcomes."\
"Preliminary searches were run and results reviewed to identify additional terms."
question=input(">>")

while question!='stop':
    answer = answer_question(question, context)
    print(f"Bot: {answer}")
    question=input(">>")