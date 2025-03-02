import os
import random
import math
import warnings

import torch
from transformers import (RobertaTokenizerFast, RobertaConfig, RobertaForMaskedLM,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from datasets import Dataset

# Suppress warnings and logs for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#######################################
# 1. Define a tiny bilingual corpus
#######################################
english_corpus = [
    """"Birds animals are a group of warm-blooded vertebrates constituting the class Aves (Latin: [ˈaveːs]), characterised by feathers, toothless beaked jaws, the laying of hard-shelled eggs, a high metabolic rate, a four-chambered heart, and a strong yet lightweight skeleton. Birds live worldwide and range in size from the 5.5 cm (2.2 in) bee hummingbird to the 2.8 m (9 ft 2 in) common ostrich. There are over 11,000 living species and they are split into 44 orders. More than half are passerine or "perching" birds. Birds have wings whose development varies according to species; the only known groups without wings are the extinct moa and elephant birds. Wings, which are modified forelimbs, gave birds the ability to fly, although further evolution has led to the loss of flight in some birds, including ratites, penguins, and diverse endemic island species. The digestive and respiratory systems of birds are also uniquely adapted for flight. Some bird species of aquatic environments, particularly seabirds and some waterbirds, have further evolved for swimming. The study of birds is called ornithology.

Birds are feathered theropod dinosaurs and constitute the only known living dinosaurs. Likewise, birds are considered reptiles in the modern cladistic sense of the term, and their closest living relatives are the crocodilians. Birds are descendants of the primitive avialans (whose members include Archaeopteryx) which first appeared during the Late Jurassic. According to some estimates, modern birds (Neornithes) evolved in the Late Cretaceous or between the Early and Late Cretaceous (100 Ma) and diversified dramatically around the time of the Cretaceous–Paleogene extinction event 66 million years ago, which killed off the pterosaurs and all non-ornithuran dinosaurs.[4][5]

Many social species preserve knowledge across generations (culture). Birds are social, communicating with visual signals, calls, and songs, and participating in such behaviour as cooperative breeding and hunting, flocking, and mobbing of predators. The vast majority of bird species are socially (but not necessarily sexually) monogamous, usually for one breeding season at a time, sometimes for years, and rarely for life. Other species have breeding systems that are polygynous (one male with many females) or, rarely, polyandrous (one female with many males). Birds produce offspring by laying eggs which are fertilised through sexual reproduction. They are usually laid in a nest and incubated by the parents. Most birds have an extended period of parental care after hatching.

Many species of birds are economically important as food for human consumption and raw material in manufacturing, with domesticated and undomesticated birds being important sources of eggs, meat, and feathers. Songbirds, parrots, and other species are popular as pets. Guano (bird excrement) is harvested for use as a fertiliser. Birds figure throughout human culture. About 120 to 130 species have become extinct due to human activity since the 17th century, and hundreds more before then. Human activity threatens about 1,200 bird species with extinction, though efforts are underway to protect them. Recreational birdwatching is an important part of the ecotourism industry.The first classification of birds was developed by Francis Willughby and John Ray in their 1676 volume Ornithologiae.[6] Carl Linnaeus modified that work in 1758 to devise the taxonomic classification system currently in use.[7] Birds are categorised as the biological class Aves in Linnaean taxonomy. Phylogenetic taxonomy places Aves in the clade Theropoda as an infraclass[8] or more recently a subclass[9] or class.
Aves and a sister group, the order Crocodilia, contain the only living representatives of the reptile clade Archosauria. During the late 1990s, Aves was most commonly defined phylogenetically as all descendants of the most recent common ancestor of modern birds and Archaeopteryx lithographica.[10] However, an earlier definition proposed by Jacques Gauthier gained wide currency in the 21st century, and is used by many scientists including adherents to the PhyloCode. Gauthier defined Aves to include only the crown group of the set of modern birds. This was done by excluding most groups known only from fossils, and assigning them, instead, to the broader group Avialae,[11] on the principle that a clade based on extant species should be limited to those extant species and their closest extinct relatives.[11]

Gauthier and de Queiroz identified four different definitions for the same biological name "Aves", which is a problem.[12] The authors proposed to reserve the term Aves only for the crown group consisting of the last common ancestor of all living birds and all of its descendants,[12] which corresponds to meaning number 4 below. They assigned other names to the other groupsUnder the fourth definition Archaeopteryx, traditionally considered one of the earliest members of Aves, is removed from this group, becoming a non-avian dinosaur instead. These proposals have been adopted by many researchers in the field of palaeontology and bird evolution, though the exact definitions applied have been inconsistent. Avialae, initially proposed to replace the traditional fossil content of Aves, is often used synonymously with the vernacular term "bird" by these researchers.[13]Most researchers define Avialae as branch-based clade, though definitions vary. Many authors have used a definition similar to "all theropods closer to birds than to Deinonychus",[15][16] with Troodon being sometimes added as a second external specifier in case it is closer to birds than to Deinonychus.[17] Avialae is also occasionally defined as an apomorphy-based clade (that is, one based on physical characteristics). Jacques Gauthier, who named Avialae in 1986, re-defined it in 2001 as all dinosaurs that possessed feathered wings used in flapping flight, and the birds that descended from them.[12][18]

Despite being currently one of the most widely used, the crown-group definition of Aves has been criticised by some researchers. Lee and Spencer (1997) argued that, contrary to what Gauthier defended, this definition would not increase the stability of the clade and the exact content of Aves will always be uncertain because any defined clade (either crown or not) will have few synapomorphies distinguishing it from its closest relatives. Their alternative definition is synonymous to Avifilopluma.[19]"""
]
hindi_corpus = [
   """भारत (आधिकारिक नाम: भारत गणराज्य, अंग्रेज़ी: Republic of India, लिप्यन्तरण: रिपब्लिक ऑफ़ इंडिया) दक्षिण एशिया में स्थित भारतीय उपमहाद्वीप का सबसे बड़ा देश है। भारत भौगोलिक दृष्टि से विश्व का सातवाँ सबसे बड़ा देश है, जबकि जनसंख्या के दृष्टिकोण से दुनिया का सबसे बड़ा देश है[20]। भारत के पश्चिम में पाकिस्तान, उत्तर पश्चिम में अफगानिस्तान, उत्तर-पूर्व में चीन, नेपाल और भूटान, पूर्व में बांग्लादेश और म्यान्मार स्थित हैं। हिंद महासागर में इसके दक्षिण पश्चिम में मालदीव, दक्षिण में श्रीलंका और दक्षिण-पूर्व में इंडोनेशिया से भारत की सामुद्रिक सीमा लगती है। इसके उत्तर में हिमालय पर्वत तथा दक्षिण में भारतीय महासागर स्थित है। दक्षिण-पूर्व में बंगाल की खाड़ी तथा पश्चिम में अरब सागर है।
1,200 ईसा पूर्व संस्कृत भाषा संपूर्ण भारतीय उपमहाद्वीप में फैली हुए थी और तब तक यहां पर हिंदू धर्म का उद्धव हो चुका था और ऋग्वेद की रचना भी हो चुकी थी।[21] इसी समय बौद्ध एवं जैन धर्म उत्पन्न हो रहे होते थे।[22] प्रारंभिक राजनीतिक एकत्रीकरण ने गंगा बेसिन में स्थित मौर्य और गुप्त साम्राज्यों को जन्म दिया।[23] उनका समाज विस्तृत सृजनशीलता से भरा हुआ था। [24]
प्रारंभिक मध्ययुगीन काल में, ईसाई धर्म, इस्लाम, यहूदी धर्म और पारसी धर्म ने भारत के दक्षिणी और पश्चिमी तटों पर जड़ें जमा लीं।[25] मध्य एशिया से मुस्लिम सेनाओं ने भारत के उत्तरी मैदानों पर लगातार अत्याचार किया,[26] अंततः दिल्ली सल्तनत की स्थापना हुई और उत्तर भारत को मध्यकालीन इस्लाम साम्राज्य में मिला लिया गया।[27] 15 वीं शताब्दी में, विजयनगर साम्राज्य ने दक्षिण भारत में एक लंबे समय तक चलने वाली समग्र हिंदू संस्कृति बनाई।[28] पंजाब में सिख धर्म की स्थापना हुई।[29] धीरे-धीरे ब्रिटिश ईस्ट इंडिया कंपनी के शासन का विस्तार हुआ, जिसने भारत को औपनिवेशिक अर्थव्यवस्था में बदल दिया तथा अपनी संप्रभुता को भी मज़बूत किया।[30] ब्रिटिश राज शासन १८५८ में शुरू हुआ। धीरे धीरे एक प्रभावशाली राष्ट्रवादी आंदोलन शुरू हुआ जिसे अहिंसक विरोध के लिए जाना गया और ब्रिटिश शासन को समाप्त करने का प्रमुख कारक बन गया।[31] 1947 में ब्रिटिश भारतीय साम्राज्य को दो स्वतंत्र प्रभुत्वों में विभाजित किया गया, भारतीय अधिराज्य तथा पाकिस्तान अधिराज्य, जिन्हें धर्म के आधार पर विभाजित किया गया।[32][33]
१९५० से भारत एक संघीय गणराज्य है। भारत की जनसंख्या १९५१ में ३६.१ करोड़ से बढ़कर २०११ में १२१.१ करोड़, २०२१ में बढ़कर १४०.७६ करोड़ हो गई।[34] प्रति व्यक्ति आय $64 से बढ़कर $1,498, 2020- 21 में $2150, शुरुआती 2022- 23 में $2450 हो गई और इसकी साक्षरता दर 16.6% से बढ़कर 74.04% पुरुषों के लिए एवम 64% महिलाओं की हो गई। भारत एक तेज़ी से बढ़ती हुई प्रमुख अर्थव्यवस्था और सूचना प्रौद्योगिकी सेवाओं का केंद्र बन गया है।[35] अंतरिक्ष क्षेत्र में भारत ने उल्लेखनीय तथा अद्वितीय प्रगति की। भारतीय फ़िल्में, संगीत और आध्यात्मिक शिक्षा वैश्विक संस्कृति में विशेष भूमिका निभाती हैं।[36] भारत ने ग़रीबी दर को काफ़ी हद तक कम कर दिया है।[37] भारत देश परमाणु बम रखने वाला देश है। भारत-चीन सीमा पर भारत का चीन से विवाद चल रहा है। कश्मीर क्षेत्र को लेकर भारत और पाकिस्तान में विवाद है।[38] लैंगिक असमानता, बाल शोषण, बाल कुपोषण,[39] ग़रीबी, भ्रष्टाचार, प्रदूषण इत्यादि भारत के सामने प्रमुख चुनौतियाँ है।[40] 21.4% क्षेत्र पर वन है।[41] भारत के वन्यजीव, जिन्हें परंपरागत रूप से भारत की संस्कृति में सहिष्णुता के साथ देखा गया है,[42] इन जंगलों और अन्य जगहों पर संरक्षित आवासों में निवास करते हैं। 12 फ़रवरी वर्ष 1948 में महात्मा गाँधी के अस्थि कलश जिन 12 तटों पर विसर्जित किए गए थे, त्रिमोहिनी संगम भी उनमें से एक है |भारत के दो आधिकारिक नाम हैं- हिंदी में भारत और अंग्रेज़ी में इंडिया (India)। इंडिया नाम की उत्पत्ति सिंधु नदी के अंग्रेज़ी नाम "इंडस" से हुई है।[43] श्रीमद्भागवत महापुराण में वर्णित एक कथा के अनुसार भारत नाम मनु के वंशज तथा ऋषभदेव के सबसे बड़े बेटे एक प्राचीन सम्राट भरत के नाम से लिया गया है। [44][45][46][47][48][49] एक व्युत्पत्ति के अनुसार भारत (भा + रत) शब्द का मतलब है आंतरिक प्रकाश या विदेक-रूपी प्रकाश में लीन। एक तीसरा नाम हिंदुस्तान भी है जिसका अर्थ हिंद की भूमि, यह नाम विशेषकर अरब तथा ईरान में प्रचलित हुआ। [50] [51] बहुत पहले भारत का एक मुंहबोला नाम सोने की चिड़िया भी प्रचलित था। [52] संस्कृत महाकाव्य महाभारत में वर्णित है की वर्तमान उत्तर भारत का क्षेत्र भारत के अंतर्गत आता था। भारत को कई अन्य नामों इंडिया, भारतवर्ष, आर्यावर्त, हिंद, हिंदुस्तान, जंबूद्वीप आदि से भी जाना जाता है।
प्राचीन भारत


(शीर्ष) ऋग्वेद की एक मध्यकालीन पांडुलिपि, मौखिक रूप से, १५००-१२०० ई.पू. (नीचे) संस्कृत महाकाव्य रामायण की एक पांडुलिपि से एक चित्रण
लगभग 55,000 वर्ष पहले (प्राचीन भारत) [53] आधुनिक मानव या होमो सेपियंस अफ्रीका से भारतीय उपमहाद्वीप में पहुँचे थे। [54][55][56] दक्षिण एशिया में ज्ञात मानव का प्राचीनतम अवशेष 30,000 वर्ष पुराना है।[57] भीमबेटका, मध्य प्रदेश की गुफाएँ भारत में मानव जीवन का प्राचीनतम प्रमाण हैं जो आज भी देखने को मिलता है। प्रथम स्थाई बस्तियों ने 9000 वर्ष पूर्व स्वरुप लिया था। 6,500 ईसा पूर्व तक आते आते मनुष्य ने खेती करना, जानवरों को पालना तथा घरों का निर्माण करना शुरू कर दिया था, जिसका अवशेष मेहरगढ़ में मिला था जो कि अभी पाकिस्तान में है।  यह धीरे-धीरे सिंधु घाटी सभ्यता के रूप में विकसित हुए,[59][58] जो की दक्षिण एशिया की सबसे प्राचीन शहरी सभ्यता है। यह 2600 ईसा पूर्व और 1900 ईसा पूर्व के मध्य अपने चरम पर थी।[61] यह वर्तमान पश्चिम भारत तथा पाकिस्तान में स्थित है।[62] यह मोहनजोदड़ो, हड़प्पा, धोलावीरा, और कालीबंगा जैसे शहरों के आसपास केंद्रित थी और विभिन्न प्रकार के निर्वाह पर निर्भर थी, यहाँ व्यापक बाजार था तथा शिल्प उत्पादन होता था।[60]
2000 से 500 ईसा पूर्व तक ताम्र पाषाण युग संस्कृति से लौह युग का आगमन हुआ। इसी युग को हिंदू धर्म से जुड़े प्राचीनतम धर्म ग्रंथ, वेदों का रचनाकाल माना जाता है तथा पंजाब तथा गंगा के ऊपरी मैदानी क्षेत्र को वैदिक संस्कृति का निवास स्थान माना जाता है।] कुछ इतिहासकारों का मानना है की इसी युग में उत्तर-पश्चिम से भारतीय-आर्यन का आगमन हुआ था।[64] इसी अवधि में जाति प्रथा भी प्रारंभ हुई थी।

"""
]
raw_corpus = english_corpus + hindi_corpus

#######################################
# 2. Train a custom ByteLevel BPE Tokenizer
#######################################
from tokenizers import ByteLevelBPETokenizer

# Write the corpus to a temporary file
corpus_file = "combined_corpus.txt"
with open(corpus_file, "w", encoding="utf-8") as f:
    for line in raw_corpus:
        f.write(line.strip() + "\n")

# Initialize and train the tokenizer
bpe_tokenizer = ByteLevelBPETokenizer()
bpe_tokenizer.train(files=corpus_file, vocab_size=8000, min_frequency=2,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
# Save the tokenizer files
tokenizer_dir = "./my_byte_bpe"
os.makedirs(tokenizer_dir, exist_ok=True)
bpe_tokenizer.save_model(tokenizer_dir)

#######################################
# 3. Create a Hugging Face Tokenizer
#######################################
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_dir, max_length=512)

#######################################
# 4. Prepare the Dataset for MLM
#######################################
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

# Create a dataset from the raw corpus
data = [{"text": line} for line in raw_corpus]
dataset = Dataset.from_list(data)
dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

#######################################
# 5. Create a Model Config and Initialize Model
#######################################
config = RobertaConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=514,
    num_attention_heads=6,
    num_hidden_layers=6,
    hidden_size=384,
    intermediate_size=1024,
    type_vocab_size=1,
)
model = RobertaForMaskedLM(config=config)

#######################################
# 6. Data Collator for Masked Language Modeling
#######################################
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

#######################################
# 7. Training Arguments and Trainer
#######################################
training_args = TrainingArguments(
    output_dir="./scratch_xlmr",
    overwrite_output_dir=True,
    num_train_epochs=5,             # Increase epochs for a real training scenario
    per_device_train_batch_size=2,
    save_steps=1000,
    save_total_limit=1,
    logging_steps=50,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

#######################################
# 8. Train and Save the Model
#######################################
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./final_xlmr_bilingual")
    tokenizer.save_pretrained("./final_xlmr_bilingual")
    print("Training complete. Model and tokenizer saved to ./final_xlmr_bilingual")
