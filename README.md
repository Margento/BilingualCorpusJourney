# BilingualCorpusJourney
A Bilingual Corpus Journey in Vector Prosody

Margento

Cannes. A Bilingual Corpus Journey in Vector Prosody

—Topographical poems translated into network topology poetry—

A journey to Cannes in the summer of 2020, in the middle of the corona pandemic, occasioned an accumulation of poems in English and French. Poems on the destination, the Riviera, and the Mediterranean (a search that soon actually morphed into a French-poems-on-the-sea search), as well as poems amassed from other previous ‘real-world’ or digital space journeys: the CROWD Tour that crossed Europe from the Arctic circle to the Mediterranean in 2016, Publishing Sphere 2018 (Montreal), DHSI 2019 and 2020 (Victoria, B.C.), alongside more recent or recurrent readings worked their way into the collection. 

The resulting bilingual corpus was then processed in its entirety by computing vectors for each poem based off of the FastText multilingual pre-trained word embeddings (trained on Common Crawl and Wikipedia) and repurposing the code written by Babylonhealth (https://github.com/babylonhealth/fastText_multilingual/blob/master/fasttext.py) for bilingual—English and French—applications. 

Then the poem vectors are correlated and the bilingual corpus is translated into a network in which the correlations between the vectors become the weights of the edges connecting the poem-nodes. As the journey was imprinted by switching between English and French all the time, from all the edges in the network only the ones connecting English to French poems are selected (round 40,000 cross-lingual edges). 

We want to trace a route progressing from the strongest English-French correlation in the network to the weakest one (while one can of course choose to go the other way round as well) and generating a poem that best describes that cross-lingual journey across the corpus. And we want to come up with a machine-based prosody in the process. 

There are perhaps three main possible ‘best’ ways to do all of the above, but only one of them is presented here (while the others will make the subject of two subsequent installments). 

At this point therefore, the poem describing the journey will be assembled by selecting from each of the poems on the route the most representative line. Since the machine reads both the poems and their lines as vectors, a most representative line will be one whose vector is the closest to the one of the whole poem it is a part of. Such algorithmic reading correlates in fact each line to all the other lines as well as to the respective poem in its entirety, which can constitute the foundation for the poetic form we need. The meter underlying this form is defined by the vector-based quantification of each line of verse. The term I am advancing for such poetic meter is vector prosody. 

The algorithm will not only select a certain line from each poem, but also certain poems from the corpus, and will be doing the latter according to a couple of constraints. As we want the route not to hit any ‘spot’ (poem-node) more than once—the foundational Euler walk problem, yet without the cross-ALL-bridges-in-Koenigsberg constraint—the journey will reach its end the moment all neighbors of a poem across the language barrier have already been visited. Also, since there are twice as many English poems as French in the corpus (outrageous indeed, and yet the disparity outputted by plain google searches—with no aid from dedicated online archives and collections beyond google rankings—would have been considerably greater), only half of the former, at most, will be selected to be on our itinerary. 

The poem triggered by a trip across France (to the “triggering town” of Cannes) and a simultaneous journey across corpora and languages, thus emerges in a (dually multiple) language that is neither natural nor machinic while being both. More precisely, the poem is written in natural language by way of (and enhanced by) natural language processing. Such language shapes a poetry of place, and is, at the same time, the language of a poem establishing itself—alongside its embedded corpora—as networked technological sites. For one cannot write a topographical poem nowadays that is not topological as well. Not only because topography in the (post)digital and pandemic age is always topological. But even more so since the poem is composed/generated as being impacted by, contributing to, and inhabiting the (“a-poetic technological”) network (cf. Tanasescu et al. 2020, https://bit.ly/2WVt9jR). Any new poem will be in and of itself such a network; it will be a #GraphPoem.

The script made available on GitHub [hyperlink to be added] can be used for any other bilingual dataset—as such, in English and French, or, if the data are in other languages, by importing the respective FastText word embeddings instead of the English and French ones—and can be adapted for multilingual corpora as well.

_________________________


[ NOTES: One cannot write a topographical poem nowadays that is not topological at the same time. Any poem is composed/generated as being impacted by, contributing to, and inhabiting a/the (poem) network. And any new poem will be itself such a network; it will be a graph poem. 

The language of poetry is neither natural nor machinic while being both, and thus in fact natural language by way of natural language processing. For NLP is the a priori of NL (its ‘new [computational neuroscience] unconscious’). Not the other way round.]
