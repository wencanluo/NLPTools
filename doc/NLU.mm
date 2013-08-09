<map version="1.0.0">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1368549803449" ID="ID_1617178244" LINK="Natural%20language%20understanding" MODIFIED="1368738037946" TEXT="NLU">
<node CREATED="1368117482680" ID="ID_1766666498" MODIFIED="1368118691305" POSITION="right" STYLE="bubble" TEXT="MyWork">
<node CREATED="1368117280383" ID="ID_61452485" MODIFIED="1368118691305" STYLE="bubble" TEXT="Data">
<node CREATED="1372276441177" ID="ID_726384269" MODIFIED="1372276471108" TEXT="Mechanical Turk">
<node CREATED="1372276473405" ID="ID_494426798" MODIFIED="1372276475256" TEXT="Task">
<node CREATED="1372276548672" ID="ID_110430146" MODIFIED="1372276557509" TEXT="constraints">
<node CREATED="1372276574929" ID="ID_1824226624" MODIFIED="1372276589671" TEXT="constraint the content but not the style">
<node CREATED="1372276500590" ID="ID_824149430" MODIFIED="1372276567509" TEXT="dummy words?">
<node CREATED="1372276507967" ID="ID_1132665033" MODIFIED="1372276516307" TEXT="simple the question"/>
<node CREATED="1372276516543" ID="ID_954144898" MODIFIED="1372276529044" TEXT="reduce the annotation effort"/>
</node>
</node>
</node>
</node>
<node CREATED="1372276475757" ID="ID_1957982128" MODIFIED="1372276484931" TEXT="How many turkers?"/>
<node CREATED="1372276485277" ID="ID_561595282" MODIFIED="1372276488642" TEXT="Time"/>
<node CREATED="1372276489069" ID="ID_914048452" MODIFIED="1372276494211" TEXT="How many response?"/>
<node CREATED="1372276494671" ID="ID_1260822448" MODIFIED="1372276496963" TEXT="Cost?"/>
</node>
<node CREATED="1369427156769" ID="ID_600340625" MODIFIED="1372276611791" TEXT="Simulation">
<node CREATED="1369427163525" ID="ID_403571439" MODIFIED="1369427182621" TEXT="Golden Stanstand"/>
</node>
</node>
<node CREATED="1368117307289" ID="ID_881491960" MODIFIED="1368550157020" STYLE="bubble" TEXT="Feature extractor">
<node CREATED="1368117331148" ID="ID_1550427255" MODIFIED="1368118691305" STYLE="bubble" TEXT="unigram">
<node CREATED="1368117351367" ID="ID_1438679708" MODIFIED="1368118691305" STYLE="bubble" TEXT="stop words filter"/>
<node CREATED="1368117380258" ID="ID_1785892660" MODIFIED="1368118691305" STYLE="bubble" TEXT="stemer"/>
<node CREATED="1368117692664" ID="ID_1822611398" MODIFIED="1368118691305" STYLE="bubble" TEXT="tolower"/>
<node CREATED="1368550166623" ID="ID_1063873494" MODIFIED="1368550171096" TEXT="synset">
<node CREATED="1368550276243" ID="ID_1969605672" MODIFIED="1368550281279" TEXT="extensition"/>
<node CREATED="1368550284736" ID="ID_947188678" MODIFIED="1368550287864" TEXT="replacement"/>
</node>
</node>
<node CREATED="1368117324164" ID="ID_1929420368" MODIFIED="1368118691305" STYLE="bubble" TEXT="bigram"/>
<node CREATED="1368117343258" ID="ID_1095006278" MODIFIED="1368118691305" STYLE="bubble" TEXT="pos"/>
<node CREATED="1368549972000" ID="ID_40942668" MODIFIED="1368549976708" TEXT="Parser Tree">
<node CREATED="1368549979273" ID="ID_222105306" MODIFIED="1368549985341" TEXT="Dependant Parser"/>
<node CREATED="1368549986811" ID="ID_950145206" MODIFIED="1368549990330" TEXT="Shallow Parser"/>
</node>
<node CREATED="1370464671140" ID="ID_1664256891" MODIFIED="1370464676990" TEXT="HasPerson"/>
<node CREATED="1370464677380" ID="ID_1640688843" MODIFIED="1370464679907" TEXT="HasLocation"/>
<node CREATED="1370464680235" ID="ID_148921060" MODIFIED="1370464683729" TEXT="HasSocial"/>
<node CREATED="1370464684603" ID="ID_113018729" MODIFIED="1370464697067" TEXT="HasPronoun"/>
<node CREATED="1370464707972" ID="ID_1897063097" MODIFIED="1370464712449" TEXT="HasFriend"/>
<node CREATED="1370464715507" ID="ID_1350500168" MODIFIED="1370464720920" TEXT="HasStreetName"/>
<node CREATED="1370464723104" ID="ID_1218573830" MODIFIED="1370464731871" TEXT="HasCityName"/>
<node CREATED="1375205402764" ID="ID_1325563062" MODIFIED="1375205408234" TEXT="Question Type">
<node CREATED="1375205410077" ID="ID_218507813" MODIFIED="1375205411274" TEXT="Why"/>
<node CREATED="1375205411676" ID="ID_367782246" MODIFIED="1375205412842" TEXT="What"/>
<node CREATED="1375205413356" ID="ID_925766609" MODIFIED="1375205418747" TEXT="Are"/>
<node CREATED="1375205419341" ID="ID_1132071256" MODIFIED="1375205420106" TEXT="Is"/>
<node CREATED="1375205420813" ID="ID_1721699797" MODIFIED="1375205425067" TEXT="find"/>
</node>
</node>
<node CREATED="1368117289852" ID="ID_1856243237" MODIFIED="1371753396316" STYLE="bubble" TEXT="Model">
<node CREATED="1368117651305" ID="ID_1248277913" MODIFIED="1368552678954" STYLE="bubble" TEXT="Basic Classification"/>
<node CREATED="1368552636946" ID="ID_1683419384" MODIFIED="1368552642326" TEXT="Example Based Classification"/>
<node CREATED="1368552650646" ID="ID_329371334" MODIFIED="1368552659404" TEXT="Search Based Classification"/>
<node CREATED="1368552803901" ID="ID_1319317266" MODIFIED="1368552807358" TEXT="Word Cloud"/>
<node CREATED="1368554178774" ID="ID_135906097" MODIFIED="1368554179726" TEXT="nuance">
<node CREATED="1368554180985" ID="ID_1073990150" MODIFIED="1368554185895" TEXT="Statistical Language Models"/>
</node>
<node CREATED="1368807562966" ID="ID_1230488105" MODIFIED="1368807564653" TEXT="Steps">
<node CREATED="1368807565762" ID="ID_733826663" MODIFIED="1368807574041" TEXT="1. Shallow Semantic Parsing"/>
<node CREATED="1368807575072" ID="ID_1975974236" MODIFIED="1368807587552" TEXT="2. Frame Classification">
<node CREATED="1369269290012" ID="ID_1946340743" MODIFIED="1369269306772" TEXT="TypeBased FrameClassification"/>
</node>
<node CREATED="1368807588318" ID="ID_648629515" MODIFIED="1368814814805" TEXT="3. Slot Filling">
<node CREATED="1369245925046" ID="ID_1206025078" MODIFIED="1369245931246" TEXT="Local Search">
<node CREATED="1369245932308" ID="ID_1757918887" MODIFIED="1369245936915" TEXT="Match Score"/>
</node>
<node CREATED="1375204574494" ID="ID_188397912" MODIFIED="1375204626896" TEXT="Constraint Optimization Problem">
<node CREATED="1375204780763" ID="ID_818146135" MODIFIED="1375205315862" TEXT="Objective Function"/>
<node CREATED="1375205316569" ID="ID_1140396615" MODIFIED="1375205320822" TEXT="Constraint"/>
</node>
</node>
</node>
<node CREATED="1369867525224" ID="ID_265024860" MODIFIED="1369867532974" TEXT="Sequence Labeling Model">
<node CREATED="1369867534456" ID="ID_1377141712" MODIFIED="1369867543859" TEXT="Each Slot is a label"/>
</node>
<node CREATED="1369935368994" ID="ID_553859695" MODIFIED="1369935388034" TEXT="What type of information the speaker is asking for?">
<node CREATED="1369935389611" ID="ID_1669213377" MODIFIED="1369935394109" TEXT="Location"/>
<node CREATED="1369935394844" ID="ID_1462178881" MODIFIED="1369935396640" TEXT="Activity"/>
<node CREATED="1369935397327" ID="ID_1307667220" MODIFIED="1369935406449" TEXT="Price"/>
</node>
<node CREATED="1369956555106" ID="ID_1758367584" MODIFIED="1369956569160" TEXT="Why Bag-of-Word is hard to beat?"/>
<node CREATED="1370644069811" ID="ID_917696508" MODIFIED="1370644081461" TEXT="Automatic Pattern Extraction"/>
<node CREATED="1370644264669" ID="ID_1646729389" MODIFIED="1370644281481" TEXT="Multi-level or Multi-Link Graph"/>
<node CREATED="1372114396080" ID="ID_957432396" MODIFIED="1372114410527" TEXT="Combine Sequence Labeling with FramePrediction"/>
</node>
<node CREATED="1369946965540" ID="ID_1092485053" MODIFIED="1369946969553" TEXT="Evaluation">
<node CREATED="1369946976706" ID="ID_1360596264" MODIFIED="1369946978720" TEXT="Basic Score"/>
<node CREATED="1369946970584" ID="ID_990563893" MODIFIED="1369946975769" TEXT="Weighted Score"/>
</node>
<node CREATED="1372809010869" ID="ID_807291305" MODIFIED="1372809015081" TEXT="TODO:">
<node CREATED="1372809016462" FOLDED="true" ID="ID_990745469" MODIFIED="1375898591783" TEXT="Frame Classification">
<icon BUILTIN="button_ok"/>
<node CREATED="1372809032007" ID="ID_1325895629" MODIFIED="1373323362341" TEXT="N-Best">
<icon BUILTIN="button_ok"/>
</node>
<node CREATED="1373045873485" ID="ID_1557813851" MODIFIED="1373308622124" TEXT="Combine Different Classifier using voting">
<icon BUILTIN="button_ok"/>
</node>
<node CREATED="1373486841691" ID="ID_1423425388" MODIFIED="1374106320058" TEXT="hierarchical classification">
<icon BUILTIN="button_ok"/>
</node>
</node>
<node CREATED="1372809056289" FOLDED="true" ID="ID_1632700600" MODIFIED="1375206010362" TEXT="Error Analysis">
<icon BUILTIN="button_ok"/>
<node CREATED="1372809067466" ID="ID_1896225868" MODIFIED="1373326243365" TEXT="Topic">
<icon BUILTIN="button_ok"/>
</node>
<node CREATED="1372809072949" ID="ID_1125065552" MODIFIED="1373405875816" TEXT="Slot">
<icon BUILTIN="button_ok"/>
<node CREATED="1373395943340" ID="ID_1715098241" MODIFIED="1373395946843" TEXT="Confusion Matrix"/>
</node>
<node CREATED="1372809075048" ID="ID_954791721" MODIFIED="1373326223973" TEXT="Domain">
<icon BUILTIN="button_ok"/>
</node>
</node>
<node CREATED="1372809124968" ID="ID_1329035316" MODIFIED="1373395770169" TEXT="Add Senna Features">
<icon BUILTIN="button_ok"/>
</node>
<node CREATED="1373405878850" ID="ID_1982846141" MODIFIED="1374106324005" TEXT="Replace with Dummy Words">
<icon BUILTIN="button_ok"/>
</node>
<node CREATED="1374106328424" ID="ID_354736006" MODIFIED="1374106333961" TEXT="Spoken Data Set">
<node CREATED="1374106335431" ID="ID_1929786324" MODIFIED="1374106342582" TEXT="# of speakers"/>
<node CREATED="1374106343270" ID="ID_1325447825" MODIFIED="1374106344758" TEXT="WER"/>
</node>
<node CREATED="1372809167017" ID="ID_1053079049" MODIFIED="1372809189045" TEXT="ATIS Features"/>
<node CREATED="1375919693658" ID="ID_312265149" MODIFIED="1375919698376" TEXT="Other Features">
<node CREATED="1375132773223" ID="ID_1109871286" MODIFIED="1375919773410" TEXT="FriendSearch">
<icon BUILTIN="full-1"/>
<node CREATED="1375919795083" ID="ID_1108254263" MODIFIED="1375919800968" TEXT="hasFriend"/>
<node CREATED="1375919801402" ID="ID_1186359140" MODIFIED="1375919804839" TEXT="hasPeople"/>
<node CREATED="1375919895211" ID="ID_839841394" MODIFIED="1375919900776" TEXT="PeopleAbstract"/>
<node CREATED="1375919805098" ID="ID_1359950259" MODIFIED="1375919807576" TEXT="hasLocation"/>
<node CREATED="1375919881787" ID="ID_1410680794" MODIFIED="1375919886296" TEXT="LocationLevel"/>
<node CREATED="1375919818906" ID="ID_1089438292" MODIFIED="1375919830376" TEXT="Plural/Single"/>
</node>
<node CREATED="1375132779078" ID="ID_405796799" MODIFIED="1375919776761" TEXT="FriendActivitySearch">
<icon BUILTIN="full-1"/>
<node CREATED="1375919787786" ID="ID_241186206" MODIFIED="1375919793416" TEXT="hasActivity"/>
<node CREATED="1375919873450" ID="ID_1806118278" MODIFIED="1375919878040" TEXT="ActivityLevel"/>
</node>
<node CREATED="1375132786054" ID="ID_1569552405" MODIFIED="1375919777625" TEXT="FriendLocation">
<icon BUILTIN="full-2"/>
<node CREATED="1375919904170" ID="ID_78105838" MODIFIED="1375919918136" TEXT="A/The"/>
</node>
<node CREATED="1375132792133" ID="ID_956214241" MODIFIED="1375919778329" TEXT="FriendActivity">
<icon BUILTIN="full-2"/>
</node>
<node CREATED="1375132842467" ID="ID_936660357" MODIFIED="1375897790056" TEXT="PlanRoute">
<icon BUILTIN="full-3"/>
<node CREATED="1375919935851" ID="ID_1257730647" MODIFIED="1375919947433" TEXT="Navigation"/>
<node CREATED="1375919952571" ID="ID_79397835" MODIFIED="1375919966520" TEXT="DestPlace"/>
<node CREATED="1375919960490" ID="ID_864260220" MODIFIED="1375919977960" TEXT="Fromto"/>
<node CREATED="1375919983483" ID="ID_1438292614" MODIFIED="1375919992872" TEXT="MoveFrame"/>
</node>
<node CREATED="1375132846258" ID="ID_1915297510" MODIFIED="1375919707041" TEXT="AddMidpoint">
<icon BUILTIN="full-3"/>
<node CREATED="1375920000107" ID="ID_451533648" MODIFIED="1375920002504" TEXT="Add"/>
</node>
<node CREATED="1375132849266" ID="ID_247795672" MODIFIED="1375919708697" TEXT="RemoveMidpoint">
<icon BUILTIN="full-3"/>
<node CREATED="1375920010650" ID="ID_1361342914" MODIFIED="1375920013496" TEXT="Delete"/>
<node CREATED="1375920014314" ID="ID_1179694690" MODIFIED="1375920017640" TEXT="Negative"/>
</node>
<node CREATED="1375132853746" ID="ID_416758716" MODIFIED="1375132858446" TEXT="StartNavigation"/>
<node CREATED="1375132862321" ID="ID_452183527" MODIFIED="1375132864606" TEXT="GoHome">
<node CREATED="1375920029562" ID="ID_1705414924" MODIFIED="1375920031368" TEXT="Home?"/>
</node>
<node CREATED="1375132866097" ID="ID_1357788241" MODIFIED="1375919645464" TEXT="CheckWeather">
<icon BUILTIN="full-1"/>
<node CREATED="1375920032666" ID="ID_686142215" MODIFIED="1375920045326" TEXT="RelatedWeather"/>
</node>
<node CREATED="1375132882704" ID="ID_1791724061" MODIFIED="1375132885613" TEXT="ShowMap">
<node CREATED="1375920047962" ID="ID_1887583580" MODIFIED="1375920053305" TEXT="Related to Map"/>
</node>
<node CREATED="1375132796405" ID="ID_1653478475" MODIFIED="1375919712226" TEXT="UpdateSocialStatus">
<icon BUILTIN="full-4"/>
<node CREATED="1375920093403" ID="ID_484866695" MODIFIED="1375920098776" TEXT="HasSocialApp"/>
</node>
<node CREATED="1375132885952" ID="ID_1926761894" MODIFIED="1375132889612" TEXT="Zoomin"/>
<node CREATED="1375132828451" ID="ID_1495915040" MODIFIED="1375919717721" TEXT="LocalSearch">
<icon BUILTIN="full-3"/>
<node CREATED="1375920110043" ID="ID_102289172" MODIFIED="1375920113208" TEXT="hasPOI"/>
</node>
<node CREATED="1375132832067" ID="ID_669460173" MODIFIED="1375920166218" TEXT="PropertyQuery">
<icon BUILTIN="full-5"/>
<node CREATED="1375920172586" ID="ID_968442230" MODIFIED="1375920186584" TEXT="PropertyRelationship"/>
</node>
<node CREATED="1375132890208" ID="ID_312404025" MODIFIED="1375132893292" TEXT="Zoomout"/>
<node CREATED="1375132894384" ID="ID_733208299" MODIFIED="1375132896972" TEXT="CurrentLocation"/>
<node CREATED="1375132755639" FOLDED="true" ID="ID_468502277" MODIFIED="1375919735032" TEXT="General">
<node CREATED="1375132906271" ID="ID_59322780" MODIFIED="1375132909387" TEXT="Yes"/>
<node CREATED="1375132909711" ID="ID_385620454" MODIFIED="1375132910475" TEXT="No"/>
<node CREATED="1375132911407" ID="ID_1450441648" MODIFIED="1375132913739" TEXT="Cancel"/>
</node>
<node CREATED="1375132749288" ID="ID_93860908" MODIFIED="1375919764891" TEXT="Unknown">
<node CREATED="1375132902847" ID="ID_339277665" MODIFIED="1375132904923" TEXT="unknown"/>
</node>
<node CREATED="1375920073435" ID="ID_1979942554" MODIFIED="1375920078907" TEXT="V-N Relationship"/>
</node>
<node CREATED="1375894654162" ID="ID_1283095631" MODIFIED="1375895314337" TEXT="+ Reference">
<node CREATED="1375895170137" ID="ID_867217266" MODIFIED="1375895371648" TEXT="Slot for Reference">
<node CREATED="1375205814990" ID="ID_895674879" MODIFIED="1375894600032" TEXT="Add Topic Feature (Combined with Slot)">
<icon BUILTIN="help"/>
</node>
<node CREATED="1375894894098" ID="ID_391770943" MODIFIED="1375894907001" TEXT="Word Class Model"/>
</node>
<node CREATED="1375898556313" ID="ID_95632909" MODIFIED="1375898558887" TEXT="Frame">
<node CREATED="1375898560249" ID="ID_880693145" MODIFIED="1375898579071" TEXT="Frame Matching">
<icon BUILTIN="help"/>
</node>
</node>
</node>
<node CREATED="1375895140889" ID="ID_1802600002" MODIFIED="1375895341591" TEXT="+ SR">
<node CREATED="1375894613587" ID="ID_599879770" MODIFIED="1375895155056" TEXT="Frame Prediction for Google + Vocon"/>
<node CREATED="1375894693607" ID="ID_443063253" MODIFIED="1375895160816" TEXT="Slot for Google + Vocon">
<node CREATED="1375894919300" ID="ID_82299313" MODIFIED="1375894928334" TEXT="Rescore SR Candidate"/>
<node CREATED="1375895401353" ID="ID_1420919098" MODIFIED="1375895405431" TEXT="Combine Both"/>
</node>
</node>
<node CREATED="1375908411578" ID="ID_324800507" MODIFIED="1375908429016" TEXT="Joint Model to predict Frame and Slot">
<icon BUILTIN="help"/>
</node>
</node>
</node>
<node CREATED="1368117458742" ID="ID_1793579882" MODIFIED="1368733664016" POSITION="left" STYLE="bubble" TEXT="Related Work">
<node CREATED="1368822022478" ID="ID_1243310306" LINK="http://rb-han.de.bosch.com/han/ieee/ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4430139&amp;tag=1" MODIFIED="1368822067441" TEXT="Mori2007"/>
<node CREATED="1368550758375" ID="ID_559343313" LINK="http://dspace.mit.edu/bitstream/handle/1721.1/6903/AITR-219.pdf?sequence=2" MODIFIED="1368550815976" TEXT="Bobrow, 1964">
<node CREATED="1368550916494" ID="ID_1754715296" LINK="http://en.wikipedia.org/wiki/STUDENT_(computer_program)" MODIFIED="1368550949838" TEXT="STUDENT"/>
</node>
<node CREATED="1368725675938" ID="ID_539176705" MODIFIED="1368725676532" TEXT="Raphael, 1964"/>
<node CREATED="1368550962444" ID="ID_1219215616" MODIFIED="1368550968184" TEXT="Weizenbaum, 1965">
<node CREATED="1368550975269" ID="ID_1024175086" LINK="http://en.wikipedia.org/wiki/ELIZA" MODIFIED="1368550982400" TEXT="ELIZA"/>
</node>
<node CREATED="1368725682766" ID="ID_139658250" MODIFIED="1368725684126" TEXT="Craig et al., 1966"/>
<node CREATED="1368551156394" ID="ID_384500725" MODIFIED="1368551161931" TEXT="Schank, 1969">
<node CREATED="1368551166623" ID="ID_1180444516" MODIFIED="1368551175631" TEXT="conceptual dependency theory"/>
</node>
<node CREATED="1368725689641" ID="ID_1665170445" MODIFIED="1368725690110" TEXT="Collins and Quillian, 1969"/>
<node CREATED="1368551106081" ID="ID_5028727" MODIFIED="1368725642297" TEXT="Winograd, 1971">
<node CREATED="1368551120235" ID="ID_1543582165" LINK="http://en.wikipedia.org/wiki/SHRDLU" MODIFIED="1368551125631" TEXT="SHRDLU"/>
</node>
<node CREATED="1368725722876" ID="ID_619049005" MODIFIED="1368725723547" TEXT="LUNAR">
<node CREATED="1368725729422" ID="ID_993963157" MODIFIED="1368725729782" TEXT="Woods et al., 1972"/>
</node>
<node CREATED="1368724046422" ID="ID_892044742" MODIFIED="1369087618385" TEXT="Meaning Representation">
<node CREATED="1368724206313" ID="ID_485846061" MODIFIED="1368724212782" TEXT="Overview">
<node CREATED="1368724221110" ID="ID_926280687" LINK="http://rb-han.de.bosch.com/han/springer/rd.springer.com/book/10.2991/978-94-91216-53-4/page/1" MODIFIED="1368724288532" TEXT="Ovchinnikova2012"/>
</node>
<node CREATED="1368724059219" ID="ID_402832085" MODIFIED="1368815218720" TEXT="Formal Semantics">
<node CREATED="1368724361797" ID="ID_934586926" MODIFIED="1368724363016" TEXT="Montague, 1973"/>
<node CREATED="1368724498594" ID="ID_520526226" MODIFIED="1368724508860" TEXT="Discourse Representation Theory">
<node CREATED="1368724513344" ID="ID_1241193412" MODIFIED="1368724534344" TEXT="Kamp and Reyle, 1993"/>
</node>
<node CREATED="1368724544766" ID="ID_578788326" MODIFIED="1368724545407" TEXT="Dynamic Predicate Logic">
<node CREATED="1368724554469" ID="ID_459499011" MODIFIED="1368724554938" TEXT="Groenendijk and Stokhof, 1991"/>
</node>
<node CREATED="1368724574251" ID="ID_1756235752" MODIFIED="1368724574641" TEXT="Segmented Discourse Representation Theory (SDRT)">
<node CREATED="1368724588172" ID="ID_880957491" MODIFIED="1368724588626" TEXT="Asher, 1993; Lascarides and Asher, 1993; Asher and Lascarides, 2003"/>
</node>
</node>
<node CREATED="1368724063969" FOLDED="true" ID="ID_730360206" MODIFIED="1369155444670" TEXT="Lexical Semantics">
<node CREATED="1368724626641" ID="ID_1187378203" MODIFIED="1368724630719" TEXT="Katz and Fodor, 1963; Jackendoff, 1972"/>
<node CREATED="1368724672266" ID="ID_1395186603" MODIFIED="1368724673032" TEXT="Bolinger, 1965; Vermazen, 1967; Putnam, 1975"/>
<node CREATED="1368724704501" ID="ID_1008322718" MODIFIED="1368724714735" TEXT="thematic roles">
<node CREATED="1368724730954" ID="ID_239985275" MODIFIED="1368724731360" TEXT="Fillmore (1968), Jackendoff (1987), and Dowty (1991)"/>
</node>
<node CREATED="1368724751594" ID="ID_1350475209" MODIFIED="1368724751969" TEXT="selectional preferences">
<node CREATED="1368724773813" ID="ID_1250419556" MODIFIED="1368724774391" TEXT="Chomsky (1965)"/>
<node CREATED="1368724775672" ID="ID_1032811069" MODIFIED="1368724783594" TEXT="McCawley, 1973"/>
</node>
<node CREATED="1368724798032" ID="ID_1270389326" MODIFIED="1368724798719" TEXT="Prototype theory">
<node CREATED="1368724805579" ID="ID_515611650" MODIFIED="1368724806016" TEXT="Rosch (1978)"/>
<node CREATED="1368724825454" ID="ID_1893735068" MODIFIED="1368725070672" TEXT="Overview: Riemer, 2010"/>
</node>
<node CREATED="1368724862672" ID="ID_1823528281" MODIFIED="1368725058672" TEXT="cognitive semantics">
<node CREATED="1368724883188" ID="ID_995682081" MODIFIED="1368724883719" TEXT="Langacker (1987) and Lakoff (1987)"/>
<node CREATED="1368724938672" ID="ID_278854808" MODIFIED="1368724987719" TEXT="Fillmore, 1968, Frame semantics"/>
<node CREATED="1368724963438" ID="ID_1723144400" MODIFIED="1368724975032" TEXT="Lakoff (1987) idealized cognitive models"/>
<node CREATED="1368725010766" ID="ID_1932762029" MODIFIED="1368725011172" TEXT="image schemes">
<node CREATED="1368725018360" ID="ID_1920847567" MODIFIED="1368725018688" TEXT="Langacker (1987)"/>
<node CREATED="1368725034079" ID="ID_66347127" MODIFIED="1368725034360" TEXT="Talmy, 1983, 2000"/>
</node>
<node CREATED="1368725059610" ID="ID_218423105" MODIFIED="1368725067860" TEXT="Overview: Riemer, 2010"/>
</node>
<node CREATED="1368725086188" ID="ID_747428591" MODIFIED="1368725086594" TEXT="lexical-semantic relations">
<node CREATED="1368725098688" ID="ID_1712949551" MODIFIED="1368725099422" TEXT="Cruse (1986)"/>
<node CREATED="1368725110094" ID="ID_1618475069" MODIFIED="1368728681251" TEXT="WordNet">
<node CREATED="1368725119126" ID="ID_1882920077" MODIFIED="1368725119672" TEXT="Miller et al., 1990; Miller and Fellbaum, 1991; Fellbaum, 1998b"/>
</node>
</node>
<node CREATED="1368725134969" ID="ID_929859853" MODIFIED="1368725135422" TEXT="Generative Lexicon (GL)">
<node CREATED="1368725146516" ID="ID_138629614" MODIFIED="1368725147313" TEXT="Pustejovsky (1991, 1995)"/>
<node CREATED="1368725182813" ID="ID_244619918" MODIFIED="1368725183485" TEXT="Kilgarriff (2001); Cimiano and Wenderoth (2007)"/>
</node>
</node>
<node CREATED="1368724074032" FOLDED="true" ID="ID_221914044" MODIFIED="1369155443420" TEXT="Distributional Semantics">
<node CREATED="1368725199391" FOLDED="true" ID="ID_842059526" MODIFIED="1368815310162" TEXT="distributional hypothesis">
<node CREATED="1368725215032" ID="ID_1985724433" MODIFIED="1368725217001" TEXT="&quot;words which are similar in meaning occur in similar contexts&quot;">
<node CREATED="1368725228469" ID="ID_1201076563" MODIFIED="1368725229704" TEXT="Rubenstein and Goodenough, 1965"/>
</node>
</node>
<node CREATED="1368725266813" ID="ID_608798330" MODIFIED="1368725267563" TEXT="Harris (1954, 1968)"/>
<node CREATED="1368725240422" ID="ID_1732788683" MODIFIED="1368725242485" TEXT="pointwise mutual information">
<node CREATED="1368725281094" ID="ID_892512838" MODIFIED="1368725282469" TEXT="Church and Hanks (1989)"/>
</node>
<node CREATED="1368725295407" ID="ID_407053609" MODIFIED="1368725295922" TEXT="vector space models">
<node CREATED="1368725359391" FOLDED="true" ID="ID_879449538" MODIFIED="1368815307650" TEXT="second order co-occurrence">
<node CREATED="1368725373813" ID="ID_924513132" MODIFIED="1368725376391" TEXT="Grefenstette,1994"/>
</node>
<node CREATED="1368725393172" FOLDED="true" ID="ID_696855328" MODIFIED="1368815307057" TEXT="Hyperspace Analogue to Language">
<node CREATED="1368725403282" ID="ID_909848743" MODIFIED="1368725403844" TEXT="Lund and Burgess, 1996"/>
</node>
<node CREATED="1368725409094" FOLDED="true" ID="ID_1711364643" LINK="http://en.wikipedia.org/wiki/Latent_semantic_analysis" MODIFIED="1368815304747" TEXT="Latent Semantic Analysis">
<node CREATED="1368729324422" ID="ID_531250316" MODIFIED="1368729326876" TEXT="?"/>
<node CREATED="1368725414235" ID="ID_1652182091" LINK="http://www.welchco.com/02/14/01/60/96/02/2901.HTM" MODIFIED="1368729248501" TEXT="Landauer, 2007"/>
</node>
<node CREATED="1368725420985" FOLDED="true" ID="ID_860270154" MODIFIED="1368815305683" TEXT="Topic-based vector space model">
<node CREATED="1368725428907" ID="ID_643792551" MODIFIED="1368725447594" TEXT="Kuropka and Becker, 2003"/>
</node>
<node CREATED="1368725452672" FOLDED="true" ID="ID_1636368979" MODIFIED="1368815306230" TEXT="Generalized vector space model">
<node CREATED="1368725458422" ID="ID_993399216" MODIFIED="1368725458782" TEXT="Tsatsaronis and Panagiotopoulou, 2009"/>
</node>
<node CREATED="1368725495641" FOLDED="true" ID="ID_1602826778" MODIFIED="1368815300362" TEXT="Examples">
<node CREATED="1368725516282" ID="ID_1796131165" MODIFIED="1368725516876" TEXT="modeling selectional preferences">
<node CREATED="1368725522516" ID="ID_61703986" MODIFIED="1368725523032" TEXT="Resnik, 1997; Erk et al., 2010; Schulte im Walde, 2010"/>
</node>
<node CREATED="1368725531579" ID="ID_903610063" MODIFIED="1368725532329" TEXT="learning qualia structures">
<node CREATED="1368725539704" ID="ID_1784784398" MODIFIED="1368725541001" TEXT="Lapata and Lascarides, 2003b"/>
</node>
<node CREATED="1368725563797" ID="ID_1635126246" MODIFIED="1368725566407" TEXT="lexical priming">
<node CREATED="1368725572047" ID="ID_504218143" MODIFIED="1368725572438" TEXT="Lund et al., 1995"/>
</node>
<node CREATED="1368725579797" ID="ID_145188976" MODIFIED="1368725580126" TEXT="synonym selection">
<node CREATED="1368725584782" ID="ID_702590894" MODIFIED="1368725585141" TEXT="Landauer and Dumais, 1997"/>
</node>
<node CREATED="1368725591219" ID="ID_1641790088" MODIFIED="1368725591626" TEXT="semantic similarity judgments">
<node CREATED="1368725597501" ID="ID_1100688458" MODIFIED="1368725597782" TEXT="Mcdonald and Ramscar, 2001"/>
</node>
</node>
</node>
</node>
<node CREATED="1368724097329" ID="ID_1067617437" MODIFIED="1368724104594" TEXT="Procedural Semantics"/>
<node CREATED="1368724105079" ID="ID_1430963661" MODIFIED="1368724115188" TEXT="Semantic Networks">
<node CREATED="1368725748516" ID="ID_1374924649" MODIFIED="1368725749110" TEXT="Quillian (1968)"/>
<node CREATED="1368725765126" ID="ID_421088108" MODIFIED="1368725768985" TEXT="Overview: Sowa, 1987"/>
<node CREATED="1368725786626" FOLDED="true" ID="ID_1146412177" MODIFIED="1368815295946" TEXT="conceptual dependency theory">
<node CREATED="1368725801985" ID="ID_1750192624" MODIFIED="1368725802344" TEXT="Schank (1972)"/>
<node CREATED="1368725816938" ID="ID_998578108" MODIFIED="1368725817282" TEXT="Schank, 1975; Schank and Abelson, 1977"/>
</node>
</node>
<node CREATED="1368724118251" ID="ID_49011374" MODIFIED="1368724119657" TEXT="Frames">
<node CREATED="1368551195166" ID="ID_808971068" MODIFIED="1368815621364" TEXT="augmented transition network (ATN)">
<node CREATED="1368551189207" ID="ID_1441048582" LINK="http://digital.cs.usu.edu/~vkulyukin/vkweb/teaching/cs6890/RSN_Woods.pdf" MODIFIED="1368815611205" TEXT="Woods, 1970"/>
</node>
<node CREATED="1368725825251" ID="ID_1061705080" LINK="http://aclweb.org/anthology-new/T/T75/T75-2022.pdf" MODIFIED="1368810249739" TEXT="Minsky (1975)"/>
<node CREATED="1369089419216" ID="ID_545185996" MODIFIED="1369089422712" TEXT="Fillmore, 1976"/>
<node CREATED="1368725838516" ID="ID_740315897" MODIFIED="1368815619897" TEXT="scripts, plans, and themes">
<node CREATED="1368725844344" ID="ID_551632970" MODIFIED="1368725844719" TEXT="Schank and Abelson (1977)"/>
<node CREATED="1368725871422" ID="ID_1698307560" MODIFIED="1368725871969" TEXT="Schank (1991)"/>
<node CREATED="1368725876282" ID="ID_494075057" MODIFIED="1368725897235" TEXT="Schank and Cleary (1995)"/>
</node>
<node CREATED="1368725902797" ID="ID_589082709" MODIFIED="1368725906266" TEXT="Overview: Barr (1980)"/>
<node CREATED="1369087701237" ID="ID_1090866692" MODIFIED="1369087721811" TEXT="Fillmore, 1985"/>
<node CREATED="1368725915641" FOLDED="true" ID="ID_1901379699" MODIFIED="1368815290906" TEXT="Ontological Semantics framework">
<node CREATED="1368725921297" ID="ID_65671729" MODIFIED="1368725921641" TEXT="Nirenburg and Raskin, 2004"/>
</node>
</node>
<node CREATED="1368724125063" FOLDED="true" ID="ID_1519568172" MODIFIED="1368815293325" TEXT="Logical Formulas">
<node CREATED="1368725933751" ID="ID_1722195201" MODIFIED="1368725934204" TEXT="Green and Raphael (1968)"/>
<node CREATED="1368725965844" ID="ID_400892458" MODIFIED="1368725966219" TEXT="Dahlgren et al., 1989; Bos and Markert, 2006"/>
<node CREATED="1368725974001" ID="ID_1007935426" MODIFIED="1368725974407" TEXT="Franconi (2003)"/>
</node>
</node>
<node CREATED="1368729678876" FOLDED="true" ID="ID_1890822124" MODIFIED="1369428595030" TEXT="Level of Knowledge">
<node CREATED="1368729691016" ID="ID_952485364" MODIFIED="1368729703251" TEXT="Syntactic "/>
<node CREATED="1368729704360" ID="ID_999134791" MODIFIED="1368729707985" TEXT="Logical"/>
<node CREATED="1368729708829" ID="ID_1160351261" MODIFIED="1368729727329" TEXT="Event frame and corresponding roles"/>
<node CREATED="1368729727891" ID="ID_1640945172" MODIFIED="1368729734344" TEXT="Type-of">
<node CREATED="1368729836938" ID="ID_1803193815" MODIFIED="1368729862594" TEXT="FrameNet, WordNet"/>
</node>
<node CREATED="1368729734969" ID="ID_1851458990" MODIFIED="1368729862594" TEXT="Common Sense Knowledge">
<node CREATED="1368729898094" ID="ID_515946987" LINK="http://www.cyc.com/platform/opencyc" MODIFIED="1368729948922" TEXT="OpenCyc"/>
</node>
<node CREATED="1368729747266" ID="ID_674399972" MODIFIED="1368729795735" TEXT="Event Calculus Semantics"/>
<node CREATED="1368729796360" ID="ID_175064323" MODIFIED="1368729810563" TEXT="Factual Ontology">
<node CREATED="1368729812344" ID="ID_213839012" LINK="http://www.mpi-inf.mpg.de/yago-naga/yago/" MODIFIED="1368729988766" TEXT="YAGO"/>
</node>
</node>
<node CREATED="1368730088126" FOLDED="true" ID="ID_176960655" MODIFIED="1369428598179" TEXT="Challenges">
<node CREATED="1368730102235" ID="ID_1815109189" MODIFIED="1368730107547" TEXT="Ambiguity">
<node CREATED="1368730108688" ID="ID_1971698131" MODIFIED="1368730115922" TEXT="phonological"/>
<node CREATED="1368730116360" ID="ID_1520894510" MODIFIED="1368730120126" TEXT="morphological"/>
<node CREATED="1368730124657" ID="ID_1957636037" MODIFIED="1368730128579" TEXT="lexical">
<node CREATED="1368730225969" ID="ID_1924635181" MODIFIED="1368730238876" TEXT="&quot;John went to the bank to open an account&quot;"/>
</node>
<node CREATED="1368730129094" ID="ID_80574980" MODIFIED="1368730132657" TEXT="syntactic">
<node CREATED="1368730190157" ID="ID_1756523163" MODIFIED="1368730244282" TEXT="&quot;John saw the man with a telescope&quot;"/>
</node>
<node CREATED="1368730133360" ID="ID_424930303" MODIFIED="1368730136407" TEXT="semantic"/>
</node>
<node CREATED="1368730138938" ID="ID_1058415059" MODIFIED="1368730148188" TEXT="Bridging">
<node CREATED="1368730272079" ID="ID_1159846839" MODIFIED="1368730279938" TEXT="Overview">
<node CREATED="1368730280751" ID="ID_411609239" MODIFIED="1368730288079" TEXT="Clark, 1975"/>
<node CREATED="1368730288704" ID="ID_43022298" MODIFIED="1368730303829" TEXT="Asher and lascarides, 1998"/>
</node>
<node CREATED="1368730313079" ID="ID_1421988261" MODIFIED="1368730322407" TEXT="Anaphora">
<node CREATED="1368730329985" ID="ID_977585366" MODIFIED="1368730346547" TEXT="&quot;John reads a book. The boy likes reading&quot;"/>
</node>
</node>
<node CREATED="1368730148954" ID="ID_68104013" MODIFIED="1368730154001" TEXT="Discourse relations">
<node CREATED="1368730412954" FOLDED="true" ID="ID_184140004" MODIFIED="1368815283744" TEXT="Temporal relation">
<node CREATED="1368730424219" ID="ID_293960084" MODIFIED="1368730438079" TEXT="&quot;Max stood up. John greeted him&quot;"/>
<node CREATED="1368730442501" ID="ID_613122492" MODIFIED="1368730453204" TEXT="&quot;John fell. Max pushed him&quot;"/>
</node>
</node>
<node CREATED="1368730165813" ID="ID_780590698" MODIFIED="1368730173391" TEXT="Implicit Predicates">
<node CREATED="1368730597594" FOLDED="true" ID="ID_1270141190" MODIFIED="1368815284711" TEXT="Noun compounds">
<node CREATED="1368730676032" ID="ID_1146216133" MODIFIED="1368730691969" TEXT="&quot;morning coffee vs. morning newspaper&quot;"/>
</node>
<node CREATED="1368730609313" FOLDED="true" ID="ID_1638885771" MODIFIED="1368815285242" TEXT="Possessives">
<node CREATED="1368730653407" ID="ID_1891563350" MODIFIED="1368730673516" TEXT="&quot;Shakespear&apos;s tragedy vs. Shakespear&apos;s house&quot;"/>
</node>
<node CREATED="1368730614297" FOLDED="true" ID="ID_1963862412" MODIFIED="1368815285757" TEXT="Prepositional Phrase">
<node CREATED="1368730633297" ID="ID_452494785" MODIFIED="1368730648797" TEXT="&quot;John in the house vs. John in anger&quot;"/>
</node>
</node>
<node CREATED="1368730154829" ID="ID_958859155" MODIFIED="1368730162360" TEXT="Metaphor and Mentonymy"/>
</node>
<node CREATED="1368744335982" ID="ID_1132645124" LINK="http://www.cs.cmu.edu/~ananlada/SemanticRoleLabelingASRU05.pdf" MODIFIED="1368744371023" TEXT="Tur2005"/>
<node CREATED="1368825074837" ID="ID_1243686794" LINK="http://researcher.watson.ibm.com/researcher/files/us-heq/W(5)%20DEEP%20PARSING%2006177729.pdf" MODIFIED="1368825130046" TEXT="McCord et al, 2012">
<node CREATED="1368825132385" ID="ID_1331453543" MODIFIED="1369068837262" TEXT="English Slot Grammar(ESG)">
<node CREATED="1368825143583" ID="ID_712198333" LINK="http://acl.ldc.upenn.edu/J/J80/J80-1003.pdf" MODIFIED="1368825182073" TEXT="McCord, 1980"/>
</node>
<node CREATED="1368825192507" ID="ID_1762513835" MODIFIED="1368825224525" TEXT="Predicate-Argument Structure (PAS)"/>
</node>
<node CREATED="1369166627396" ID="ID_1252130800" MODIFIED="1369166632155" TEXT="Exisited System">
<node CREATED="1369166637195" ID="ID_45196002" MODIFIED="1369166638381" TEXT="Siri">
<node CREATED="1369434112550" ID="ID_1597206182" LINK="http://www.jeffwofford.com/?p=817" MODIFIED="1369434123308" TEXT="introducation"/>
</node>
<node CREATED="1369166674945" ID="ID_362509021" MODIFIED="1369166678753" TEXT="ITSOPKE"/>
<node CREATED="1369433336726" ID="ID_1492783297" LINK="http://www.infocom-if.org/company/company.html" MODIFIED="1369433345925" TEXT="Infocom"/>
<node CREATED="1369433605967" ID="ID_1222983079" MODIFIED="1369433607588" TEXT="http://thcnet.net/zork/"/>
</node>
<node CREATED="1369430360344" ID="ID_112328523" MODIFIED="1369430374049" TEXT="Speech when Driving">
<node CREATED="1369430436147" ID="ID_1677041554" LINK="http://www.umich.edu/~driving/documents/TRL_guidelines.pdf" MODIFIED="1369430446999" TEXT="Design Guildlines"/>
</node>
<node CREATED="1369431603397" ID="ID_45066612" MODIFIED="1369431609508" TEXT="Natural Language Interface">
<node CREATED="1369431611286" ID="ID_1246265014" MODIFIED="1369431613921" TEXT="DataBase">
<node CREATED="1369431626658" ID="ID_1015728250" MODIFIED="1369431657653" TEXT="Ritchie, 1995"/>
</node>
</node>
<node CREATED="1371753397946" ID="ID_1612683593" MODIFIED="1371753400021" TEXT="Baseline">
<node CREATED="1369865648346" ID="ID_848379223" LINK="http://www2.research.att.com/~fsmtools/fsm/" MODIFIED="1371772481841" TEXT="FST"/>
<node CREATED="1369865834548" ID="ID_1626398678" LINK="http://www.chasen.org/~taku/software/yamcha/" MODIFIED="1369866683226" TEXT="Yamcha"/>
<node CREATED="1369865838585" ID="ID_1923969730" MODIFIED="1369865840798" TEXT="CRF"/>
<node CREATED="1369960129030" ID="ID_640782228" LINK="http://eprints-phd.biblio.unitn.it/280/1/PhD-Thesis-Dinarelli.pdf" MODIFIED="1369960192629" TEXT="Dinarelli2010-PhD-Thesis"/>
</node>
</node>
</node>
</map>
