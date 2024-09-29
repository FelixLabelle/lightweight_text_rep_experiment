from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Pool
import json

from gensim.utils import tokenize
import numpy as np
from scipy.stats import permutation_test
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm

TAG = "v10"
MAX_ITERS = 10
verbose = False
MAX_NGRAM_SIZE = 2

class DupeIdxCounterVectorizer(CountVectorizer):
    def _validate_vocabulary(self):
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                # Ignores checks on vocab integrity that would fail duped idxs
                pass
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False
        
def my_tokenize(text):
    return list(tokenize(text,lower=True))

def train_model(train_features, val_features, train_labels, val_labels, random_state):
    clf = MLPClassifier(hidden_layer_sizes=(10,100,10),random_state=random_state, max_iter=MAX_ITERS,verbose=verbose).fit(train_features, train_labels)
    val_preds = clf.predict(val_features)
    return classification_report(val_labels, val_preds, output_dict=True)

def args_train_model(args):
    train_features, val_features, train_labels, val_labels, random_state = args
    return train_model(train_features, val_features, train_labels, val_labels, random_state)
    
if __name__ == "__main__":
    config = json.load( open(f"wiki_eng_hash_similarity_{TAG}.json","r"))
    labels = config['clusters']
    n_grams =  config['list_n_grams']
    clusters = defaultdict(list)
    
    for label,n_gram in tqdm(zip(labels,n_grams)):
        clusters[label].append(n_gram)
    #import pdb;pdb.set_trace()
    normal_vocab = {n_gram : idx for idx,n_gram in enumerate(n_grams)}
    clustered_vocab = {n_gram: label for label,label_n_grams in clusters.items() for n_gram in label_n_grams}
    
    normal_featurizer = DupeIdxCounterVectorizer(tokenizer=my_tokenize,vocabulary=normal_vocab,ngram_range=(1,MAX_NGRAM_SIZE))
    clustered_featurizer = DupeIdxCounterVectorizer(tokenizer=my_tokenize,vocabulary=clustered_vocab,ngram_range=(1,MAX_NGRAM_SIZE))
    
    seeds = [369589461, 186158379, 1996187662, 3261350419, 1161594693, 3841799204, 2766708526, 511233129, 2344249448, 1473791957, 1719868133, 2879438446, 3262465858, 4054908075, 3572137546, 372058695, 2288543391, 102386023, 1989927692, 4064603696, 883250586, 2053475905, 2828099129, 1352399896, 2158090352, 1250236077, 2353360284, 3666386626, 4098522316, 3277667392, 3597889417, 127984505, 1291818164, 3261807461, 1523578656, 1165322814, 1347256985, 108286645, 3069560180, 2552683849, 1501612411, 4239435913, 17333582, 1837328441, 3341457257, 4079779970, 1981642577, 2074636585, 1306049829, 928325326, 400646631, 2203586760, 3223722163, 3424171338, 1430213302, 2421328485, 1194635892, 3137678108, 2991731667, 1174461272, 4208609829, 2236017307, 1780576147, 2799637130, 746182491, 337503451, 155260868, 2639874415, 969437701, 164419819, 3207945758, 532876100, 4098369894, 2988955460, 1765415581, 4147598888, 3923040464, 3229611038, 3748398798, 469220913, 4108501181, 3053728571, 3636705548, 3632642977, 3423323996, 2120695151, 2966883054, 3265161050, 1465333913, 3640253150, 1912409721, 3074121645, 348716213, 1385658930, 368816840, 353837276, 1045439711, 545485629, 2285600504, 949543977, 4252513287, 2387613444, 2832795374, 2119805596, 2261256900, 334832882, 3277344246, 2211436442, 830504890, 227773533, 2648219988, 1217625163, 2528944306, 1454166196, 1501037877, 1195603212, 1096009576, 1848917138, 3378062955, 4112349987, 2155711122, 3261162102, 3939253405, 1238688093, 4088482404, 1267423590, 2627145445, 769193287, 1534903034, 2935360118, 2608954739, 1568894012, 2674562113, 4279019946, 4207272290, 3227977576, 1003987979, 702252663, 2473134039, 3618336539, 3921959177, 3361544046, 3616529081, 267430105, 1373019555, 3889953686, 2917341671, 623673098, 2693993289, 600637881, 755178522, 2997700317, 4027491012, 3217954181, 675209856, 454383524, 137433291, 803817628, 2214828007, 2873181393, 428267920, 42648985, 244806243, 2113807734, 3950653754, 489977415, 470662280, 1957934054, 3284610553, 2054072725, 1094850568, 1663894762, 3886462950, 1891442800, 1985231963, 1657732627, 1893062133, 3154674937, 1558763599, 2300468136, 3138107689, 1575925056, 3355980722, 2669498073, 4259361132, 817714830, 2801557860, 2866897185, 2611435697, 1207755333, 3011188205, 3860422785, 2187127796, 12267526, 2404441610, 3665850474, 1027612041, 1989305989, 1318011216, 3306186382, 4276861943, 1057866753, 829060861, 3927906719, 3711057781, 3858715697, 4012799712, 801294879, 798694034, 2124085598, 1184647165, 3541668843, 3600685701, 713397791, 2386836495, 272725996, 636425106, 777281362, 552447285, 3250189995, 891865604, 153720715, 4201915889, 2803394712, 3258358603, 4203559411, 3982277160, 2909502610, 3412815313, 936487508, 33626785, 567736073, 4004270522, 1255623426, 2157742686, 574400891, 3941538874, 3735130956, 3566895940, 2347734551, 1203245085, 1830631584, 1517921057, 129406958, 4072999989, 1161383030, 3317211391, 4054871019, 2430646437, 3614934058, 3064578159, 3482606056, 643605072, 1652036800, 158448031, 2954147482, 359587889, 698632827, 4149537559, 1272777040, 1701908370, 484179673, 3422607076, 527449155, 4224845484, 2204837369, 1390502880, 3140010031, 2196936776, 748539217, 3692079543, 3833926903, 2249910000, 3917363089, 1257516779, 2151674528, 84403825, 1402765881, 1232540069, 3131925662, 1676104057, 3477217645, 3905520356, 2703667815, 436186378, 3571811927, 2418833913, 3339234562, 3460414312, 2572234832, 903489556, 1796035820, 3920816117, 1765041101, 824243660, 1120566096, 3450236045, 2523429241, 1981658077, 809616759, 2826043774, 3497318044, 3584468938, 4291109288, 1580319044, 339572652, 962999658, 3938176403, 935585826, 1136803461, 2986140495, 3037333951, 4196869431, 3351290647, 3649197983, 262814644, 3090425242, 665289733, 3608865497, 2119709773, 3796668455, 4227677321, 3695756620, 1355348978, 2018065096, 848094177, 2554751173, 1895674379, 1420710723, 2491329959, 3193782743, 1423668288, 373750856, 501555444, 3550104593, 1496693602, 3740827813, 180805816, 1498788597, 439727758, 95949387, 4259671444, 1559616190, 1135539028, 3442855008, 3833985344, 2329129494, 3206457852, 1666085483, 2025455567, 3180314393, 582833720, 1554152422, 228781640, 2382838611, 3435038486, 2404707633, 3725823069, 2190086221, 4142229562, 4095372549, 2576913668, 948725171, 2216342983, 3462736346, 36849933, 3466455092, 2345512000, 4154911096, 2160236123, 4232461051, 3355438168, 617716120, 3157043570, 845657926, 4024121180, 2497897689, 61619768, 1833146860, 3459391612, 532706538, 3426101796, 3556677802, 4158629642, 2688822199, 503290210, 1846995948, 3929639615, 3561932272, 3233248221, 2931607161, 482109004, 3582222226, 966967832, 3161663314, 3681863385, 3596924424, 1297025946, 370923306, 2204122878, 3116854737, 3108571606, 1147159527, 872832977, 314555770, 2718535525, 3427270579, 4208922588, 2102546294, 1751069834, 2013702767, 90656525, 3354074524, 307344616, 1883691229, 2549794159, 977186156, 1993380162, 3164782608, 305565003, 557436063, 2902019000, 1087285741, 1989781402, 3237975634, 1176870506, 1336309535, 466352705, 276793995, 2196140013, 1623911135, 3870230068, 3599017538, 1108618072, 2726528382, 1968662295, 1572232625, 3756452644, 2886439238, 3430836358, 1157013194, 4157836156, 3868377027, 1651742828, 541398163, 184716458, 3446641258, 2422490577, 2287793036, 2870005036, 2972905361, 1995133983, 3778384586, 3310306343, 1257141443, 2293699245, 3143120316, 1169245125, 3329171441, 3874745319, 3006478172, 2702564001, 3619823586, 2524266706, 3637756482, 356270319, 1527499230, 3496512666, 3841094588, 1104618129, 1274709871, 678444904, 180491754, 813931306, 2212170734, 135296229, 3502097451, 2647677908, 1981313722, 1622671817, 282841343, 3367969336, 1271022609, 2070452879, 509269822, 1624155946, 1321125871, 3482129101, 3112968163, 2054129552, 2308054789, 1737215637, 1237022899, 1737661073, 1118164240, 2671642624, 3515559447, 1782480624, 3281721066, 4141266292, 1545793138, 1500124723, 329015712, 3156997316, 3930257516, 1694308825, 2660712435, 2391625411, 1300091192, 2705772995, 1789930119, 4171943839, 3687788494, 1281339287, 3137138654, 3379480460, 1356203015, 3347002039, 3002994072, 147017196, 3913986789, 2196653498, 1709204152, 2407720249, 107644251, 3971174102, 1719237622, 1293053118, 2769445255, 3713464226, 2246333690, 1514708183, 948469436, 1774903673, 3923587550, 690735442, 223528091, 4217656974, 3789846006, 558137599, 3906123144, 2885818662, 3945166735, 762923251, 835681946, 3941264845, 2771069253, 4628409, 2574472795, 3237763114, 3907087998, 4102230014, 2164782399, 3492605852, 2853443123, 1500083869, 2940564597, 3283581529, 2036222098, 2497587361, 4009889781, 16348258, 776973955, 3759607223, 3958565813, 1590852250, 3758928959, 2344083017, 2126665588, 99049815, 1793471585, 3655004628, 3630240170, 3562881880, 893316239, 1668124562, 21083693, 185240582, 3633196684, 181711577, 3126607473, 3128823600, 952602995, 430313090, 1690944143, 3046314165, 3812823295, 4016545352, 3813009603, 3773055305, 1626747859, 3671109765, 1244105247, 3011453859, 2173532984, 2556849568, 1520410952, 1602221962, 1805732092, 2657996281]
    #[316954464, 1433699542, 233229003, 2169536044, 3079226637, 2098570363, 4071498956, 147632971, 2840389407, 1371054887, 2846185110, 1976637316, 3198864199, 1148021075, 2642432764, 2700933996, 70964088, 765298535, 3675092719, 2108329166, 2011541707, 2078429042, 4233718348, 1463765429, 2247093879, 2757204146, 2028824371, 261555554, 670977322, 412552843, 580742065, 412989141, 2249389331, 3983116367, 1774936177, 1563535200, 42023605, 2617901261, 1987696311, 2286466253, 2593982633, 4244729846, 2087664158, 3388478751, 3662919667, 1909149230, 3529540157, 4281190781, 3627994478, 971307197, 1242261286, 1385928075, 3174752292, 2354500466, 2815723820, 533511313, 2468217621, 4079156725, 1632304232, 4142391236, 837973379, 954439117, 37612330, 2897654897, 2307697954, 680923753, 2088013251, 1288627407, 3725148892, 1349341940, 990859032, 1921085834, 3456345719, 2874962499, 2486245524, 2274725698, 3394150141, 377110372, 4231110624, 1992882545, 2747648030, 1313733316, 1381632043, 2481349566, 303550167, 3499502722, 508478874, 1428857810, 3019528574, 1397813345, 812834795, 3922445218, 1453606006, 525875515, 3952549731, 2519306377, 3017058924, 1470637846, 2136549213, 2098601344, 1549991920, 386373380, 1169843644, 3285183852, 982007571, 1129866688, 601947216, 2419861947, 522893264, 474077688, 2773848484, 1549671724, 1984821524, 4032779746, 2123870827, 2689199077, 2267051130, 166650700, 4087136853, 3615990851, 6539647, 1920295831, 4246414469, 2076612953, 2083285299, 469504797, 1107332228, 238155392, 2311065337, 3665459815, 3920932985, 4013359066, 3180538409, 1704512589, 822597179, 1699423910, 3603633708, 3133321537, 2149323472, 702399131, 1358191484, 4243552809, 1329662541, 3121254, 401333240, 3749038365, 339350599, 1024947722, 3150395388, 1988682390, 2471682972, 367702412, 2678304139, 2333046377, 521175097, 1999546811, 3712765700, 2526747690, 1057916088, 2931549642, 1144985232, 1601426566, 1788828104, 2038640298, 3939320727, 3637149779, 1220597744, 3096746088, 3012166933, 1320755840, 1223209975, 1826183398, 1134102198, 3641562841, 3968453468, 3519839111, 2225175975, 1088157116, 939121045, 2537676168, 1822255707, 2994519802, 358352469, 989243046, 2245305392, 2996102205, 1232249915, 3377374284, 3447316411, 2124403195, 1458802091, 1867157045, 595041624, 3818602665, 850526553, 2761162902, 1821688072, 512011422, 3653230219, 2722680362]
    train_dataset = fetch_20newsgroups(subset='train')
    val_dataset = fetch_20newsgroups(subset='test')
    featurizer_dict = {"clustered_featurizer" : clustered_featurizer, "normal_featurizer" : normal_featurizer}
    
    bow_results = defaultdict(list)
    #import pdb;pdb.set_trace()
    for featurizer_name, featurizer in tqdm(featurizer_dict.items()):
        raw_train_features = featurizer.transform(train_dataset.data)
        raw_val_features = featurizer.transform(val_dataset.data)
        train_features = normalize(raw_train_features,axis=1)
        val_features = normalize(raw_val_features,axis=1)
        
        args = [(train_features, val_features, train_dataset.target, val_dataset.target,seed) for seed in seeds]
        with Pool() as pool:
            bow_results[featurizer_name] = [result for result in tqdm(pool.imap(args_train_model, args),total=len(args))]
        '''
        for random_state in tqdm(seeds[:50]): #,2034,4294967294,7676,4234,3013,2,12344,21150745]):
            bow_results[featurizer_name].append(train_model(train_features, val_features, train_dataset.target, val_dataset.target,random_state))
        '''
    '''
    args = []
    with Pool() as pool:
        seeds = [seed for seed in tqdm(pool.imap(calculate_sim, args),total=len(args))]
    '''

    compressed_results_dict = {featurizer_name : [result['macro avg']['f1-score'] for result in results] for featurizer_name,results in bow_results.items()}
    json.dump(compressed_results_dict,open(f"bow_results_{TAG}_{MAX_NGRAM_SIZE}.json",'w'))
    import pdb;pdb.set_trace()
    
    
    def delta(*samples):
        means = np.mean(samples,axis=1)
        return means[0] - means[1]
        
    results = permutation_test([compressed_results_dict['clustered_featurizer'],compressed_results_dict['normal_featurizer']], delta,alternative="greater")
    
    from matplotlib import pyplot as plt
    
    all_f1_scores = compressed_results_dict['clustered_featurizer'] + compressed_results_dict['normal_featurizer']

    # Define 20 bins over the range of all F1 scores
    min_score = min(all_f1_scores)
    max_score = max(all_f1_scores)
    bins = np.linspace(min_score, max_score, 20)

    # Plotting the F1 Score Distributions
    plt.figure(figsize=(10, 6))

    # Clustered Featurizer
    plt.hist(compressed_results_dict['clustered_featurizer'], bins=bins, alpha=0.7, label='Clustered Featurizer', color='blue', edgecolor='black')

    # Normal Featurizer
    plt.hist(compressed_results_dict['normal_featurizer'], bins=bins, alpha=0.7, label='Normal Featurizer', color='orange', edgecolor='black')

    # Add title and labels
    plt.title('Macro-weighted F1 Score Distribution', fontsize=16)
    plt.xlabel('Macro-weighted F1 Score', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add legend
    plt.legend(loc='upper right')

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig('f1_score_plot.png')

    # Show the plot (optional, can be removed if you just want to save)
    plt.show()