"""
PRAG — the Gradient's topics: the keys of a persona's topic_weights
(biases/<account>.json "profile"), each with the feed category it falls
under, whether it's political, and angles the generator can take on it.
"""

TOPICS: dict[str, tuple[str, bool, list[str]]] = {
    # topic: (category, political, angles)
    "national_security": ("news_politics", True, [
        "a border standoff and what satellite images show", "a new indigenous fighter jet or missile test",
        "a counter-terror operation in Kashmir", "Navy ship commissioning", "Agniveer recruitment debate",
        "defence exports growth", "veterans' pensions (OROP)", "a drone incursion along the western border",
        "India–China LAC talks", "a soldier's homecoming",
    ]),
    "infrastructure": ("news_politics", True, [
        "a new expressway opening", "Vande Bharat train on a new route", "a metro line's delays",
        "bullet train progress", "a new airport terminal", "a bridge collapse and who is responsible",
        "potholes after the monsoon", "rural roads under PMGSY", "a tunnel through the Himalayas",
        "the cost of tolls for truckers",
    ]),
    "government_schemes": ("news_politics", True, [
        "how to apply for PM Awas Yojana", "Ayushman Bharat card in a hospital", "free ration scheme",
        "PM Kisan instalment date", "Ujjwala gas refills", "Jan Dhan accounts ten years on",
        "a scheme that never reached a village", "Digital India in a small town", "Startup India loans",
        "a state welfare scheme versus a central one",
    ]),
    "party_politics": ("news_politics", True, [
        "a state election result", "a rally speech clip", "a party switch by an MLA",
        "a parliament walkout", "an opposition alliance meeting", "exit polls versus results",
        "a debate show shouting match", "a politician's old video resurfacing", "campaign promises scorecard",
        "a by-election upset",
    ]),
    "economy": ("business_money", True, [
        "GDP growth numbers", "petrol and diesel prices", "inflation in the vegetable market",
        "the rupee against the dollar", "GST collections", "unemployment figures",
        "the Union Budget explained", "manufacturing under Make in India", "a new trade deal", "onion prices",
    ]),
    "labour": ("news_politics", True, [
        "gig workers' strike for delivery partners", "new labour codes", "a factory accident",
        "MGNREGA wages delayed", "contract teachers protesting", "migrant workers heading home",
        "a union wins a pay rise", "twelve-hour shift rules", "domestic workers' rights", "sanitation workers' safety",
    ]),
    "public_services": ("news_politics", True, [
        "a government school's results", "a district hospital without doctors", "water supply cuts in a city",
        "public bus fares", "a paper leak in a recruitment exam", "railway ticket refunds",
        "free electricity units", "a mohalla clinic", "school mid-day meals", "an ambulance that came late",
    ]),
    "civil_liberties": ("news_politics", True, [
        "an internet shutdown", "a journalist arrested", "a student protest at a university",
        "a bulldozer demolition", "sedition and UAPA cases", "press freedom rankings",
        "a comedian's show cancelled", "the right to protest at a city's protest site", "surveillance and privacy",
        "a court grants bail after years",
    ]),
    "environment": ("news_politics", True, [
        "Delhi's air quality index", "a heatwave and outdoor workers", "a river cleaning project",
        "trees cut for a highway", "floods in Assam", "solar parks in Rajasthan", "a glacier lake warning",
        "plastic ban enforcement", "a mining project in a forest", "an EV subsidy",
    ]),
    "crime_safety": ("news_politics", True, [
        "a cyber fraud call and how to spot it", "a road rage video", "women's safety on night buses",
        "a police encounter", "a fake loan app racket", "a chain snatching caught on CCTV",
        "a court verdict in an old case", "drunk driving checks", "a stampede at an event", "digital arrest scams",
    ]),
    "world_news": ("news_politics", True, [
        "a war in the Middle East and Indians stuck there", "the US election and India", "a G20 or BRICS summit",
        "India at the UN", "Pakistan's economic crisis", "Bangladesh politics", "a Chinese app ban",
        "Indian students in Canada", "oil prices and OPEC", "a climate summit",
    ]),
    "cricket": ("sports", False, [
        "an India–Pakistan match moment", "an IPL auction", "a young pacer's debut", "Virat Kohli's form",
        "a last-ball finish", "the women's team winning a series", "a captaincy debate", "a stadium crowd chant",
        "gully cricket", "a fielding blunder",
    ]),
    "football": ("sports", False, [
        "an ISL derby", "the India national team's ranking", "a Premier League title race",
        "Champions League night", "a local football club in Kerala or Goa", "Messi versus Ronaldo",
        "a youth academy", "a transfer deadline deal", "fan chants from Kolkata", "a penalty shootout",
    ]),
    "other_sports": ("sports", False, [
        "Neeraj Chopra's javelin throw", "a badminton final", "chess prodigies", "kabaddi league",
        "an Olympic medal hope", "a wrestler's comeback", "hockey team", "a marathon in Mumbai",
        "F1 at a new circuit", "boxing",
    ]),
    "gadgets": ("tech_science", False, [
        "a phone launch under 20,000", "an honest earbuds review", "a laptop for students",
        "a smartwatch's battery test", "an iPhone versus Android comparison", "a camera phone night-mode test",
        "fast charging myths", "a budget TV", "a gaming phone", "an unboxing",
    ]),
    "tech_ai": ("tech_science", False, [
        "an AI tool for students", "deepfakes in elections", "a chatbot in Hindi", "UPI new features",
        "jobs AI might change", "a data leak", "semiconductor fabs in India", "coding bootcamps",
        "an AI-made song", "5G in villages",
    ]),
    "science_space": ("tech_science", False, [
        "an ISRO launch", "Chandrayaan's findings", "Gaganyaan astronaut training", "a solar eclipse",
        "a new species found in the Western Ghats", "how monsoons form", "a science fair invention",
        "Aditya-L1", "a black hole image explained", "vaccines explained",
    ]),
    "hindi_comedy": ("comedy_memes", False, [
        "a family dinner skit", "a stand-up bit about Indian parents", "an office boss sketch",
        "a wedding guest skit", "a mimicry of a cricket commentator", "a school-days sketch",
        "a landlord and tenant skit", "a relatives-asking-about-marriage bit", "a train journey skit",
        "an auto-rickshaw driver sketch",
    ]),
    "satire": ("comedy_memes", True, [
        "a mock news bulletin about potholes", "a parody of a politician's speech", "a satirical ad for a scheme",
        "a meme about exam paper leaks", "a mock interview with a minister", "sarcastic 'achievements' list",
        "a parody of TV debate anchors", "a satirical song about prices", "a roast of a new policy",
        "a cartoonist's animated strip",
    ]),
    "street_food": ("food", False, [
        "Indore's Sarafa night market", "Delhi chole bhature", "Mumbai vada pav", "Kolkata phuchka",
        "a dosa cart in Bengaluru", "momos in a hill town", "Lucknow kebabs", "a hygiene check on a stall",
        "a 10-rupee meal", "a chai stall's story",
    ]),
    "indie_music": ("music_dance", False, [
        "a Bandra open mic", "an indie band's new single", "a Coke Studio style collab", "a busker in a metro",
        "a Hindi indie love song", "a lo-fi producer at home", "a music festival in Goa",
        "a protest song", "a ghazal cover", "a guitar lesson",
    ]),
    "film_tv": ("film_tv", False, [
        "a box office weekend", "a web series review", "a trailer breakdown", "a celebrity wedding",
        "a South film's Hindi dub", "an award show moment", "a biopic controversy", "an actor's interview clip",
        "OTT recommendations", "a classic song's story",
    ]),
    "photography": ("lifestyle", False, [
        "a street photography walk in Old Delhi", "phone photography tips", "a monsoon landscape",
        "a wildlife photographer at Ranthambore", "film camera revival", "editing in Lightroom",
        "a portrait series of workers", "golden hour at a ghat", "astrophotography in Ladakh",
        "a camera budget guide",
    ]),
    "gaming": ("tech_science", False, [
        "a BGMI tournament", "a GTA trailer reaction", "an Indian game studio", "a budget gaming PC build",
        "esports as a career", "a speedrun", "a retro console", "a mobile game ban",
        "a gaming streamer's milestone", "a Valorant clutch",
    ]),
    "religion": ("religion", False, [
        "an aarti at a ghat", "a temple's history", "a festival preparation", "a devotional bhajan",
        "a pilgrimage route", "an interfaith iftar", "a gurdwara langar", "a saint's teaching",
        "a temple inauguration", "a Kumbh Mela crowd",
    ]),
    "astrology": ("religion", False, [
        "this week's horoscope", "a vastu tip for your home", "a planet's transit and your sign",
        "a gemstone recommendation", "a tarot reading", "a numerology prediction", "lucky colours", "a palmistry clip",
        "a muhurat for buying gold", "a zodiac compatibility list",
    ]),
    "fashion_beauty": ("lifestyle", False, [
        "a wedding outfit haul", "a skincare routine", "a makeup tutorial", "a saree draping guide",
        "a sneaker drop", "a hair care tip", "an OOTD reel", "a budget fashion haul", "a men's grooming tip",
        "a jewellery collection",
    ]),
    "lifestyle": ("lifestyle", False, [
        "a day in the life in Bengaluru", "a couple's vlog", "a minimalist room tour", "a morning routine",
        "a pet's reaction", "a family festival vlog", "a long-distance relationship story", "a house plant tour",
        "moving to a new city", "a hostel life vlog",
    ]),
    "finance": ("business_money", False, [
        "SIP versus FD", "a stock market crash explained", "how to file income tax", "a credit card trap",
        "gold prices", "an IPO frenzy", "saving on a 30,000 salary", "crypto rules", "a home loan EMI",
        "a finance influencer's tip",
    ]),
    "history_explainers": ("education", False, [
        "the story of the Constitution", "a forgotten freedom fighter", "the Mughal and Maratha eras",
        "partition stories", "how the Green Revolution happened", "the Emergency of 1975",
        "ancient Indian science", "a fort's history", "the 1991 reforms", "the history of a city",
    ]),
    "travel_nature": ("travel_nature", False, [
        "a budget trip to Spiti", "Kerala backwaters", "a waterfall in the monsoon", "a tiger sighting",
        "a road trip on a new highway", "a hidden beach in Gokarna", "a trek in Uttarakhand", "a stray dog rescue",
        "a hill station crowd", "a train journey through the Western Ghats",
    ]),
    "promotion": ("promotion", False, [
        "a creator plugging a skincare brand", "a fantasy sports app ad", "a coaching institute ad",
        "a betting app disguised as a game", "a giveaway", "a new restaurant's paid promotion",
        "a course that promises income", "a phone brand sponsorship", "an online sale", "a protein powder plug",
    ]),
}
STANCES = ("supportive", "critical", "neutral")
FORMATS = ("news clip", "explainer", "reaction", "street interview", "skit", "meme edit", "vlog",
           "podcast clip", "review", "speech clip", "ground report")
LANGUAGES = {"en": "English", "hi": "Hindi (Devanagari)", "hinglish": "Hinglish (romanised Hindi mixed with English)"}


def category(topic: str) -> str:
    return TOPICS.get(topic, ("other", False, []))[0]


def political(topic: str) -> bool:
    return TOPICS.get(topic, ("", False, []))[1]
