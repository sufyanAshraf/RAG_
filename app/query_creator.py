import json

# from groq import Groq
# from langchain_groq import ChatGroq

class queryCreator:
    def __init__(self):
        self.categories =  ["spa", "hotel", "restaurant"]
        self.filter = ["name" , "city", "region" , "rating" , "distance" ] 
        self.hotel_services = ['Free WiFi', 'Swimming Pool', 'Gym', 'Restaurant', 'Parking', 'Room Service', 'Airport Shuttle', 'Spa', 'Breakfast', 'Garden', 'Private Dining']
        self.spa_services = ['Thai Massage', 'Swedish Massage', 'Deep Tissue Massage', 'Hot Stone Massage', 'Aromatherapy', 'Couples Massage', 'Foot Massage', 'Head Massage', 'Sports Massage', 'Facial Massage']
        self.restaurant_services = ['Biryani', 'Chicken Karahi', 'Chicken Tikka', 'Naan', 'Raita', 'Burgers', 'Pizza', 'French Fries', 'Chicken Wings', 'Pasta', 'Mutton Karahi', 'Seekh Kebab', 'Kheer', 'Sandwiches', 'Shawarma', 'Gulab Jamun', 'Chocolate Cake', 'Coffee', 'Haleem']

    def create_filter(self, response):
             
        data = json.loads(response)
    
        query_filter = {}

        field_mapping = {
            "Category": "Category",
            "city": "city",
            "region": "region",
            "rating": "rating"
            # "distance": "distance" # you need distance in integer otherwise it will do exact match
        }
    
        for json_key, db_key in field_mapping.items():
            value = data.get(json_key)
            
            if value is not None:
                query_filter[db_key] = {"$eq": value}

        # Services
        services = data.get("service") 
        
        if services:
            services = [s for s in services if s is not None]

            if services:
                query_filter["services"] = {"$in": services}


        return query_filter

    def create_query(self, model, query):
        prompt= self.prompt_create_query(query) 
        response = model.invoke_model( prompt)

        # system_prompt = "You are a helpful assistant."
        # response = model.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
        # response = response.content 
        
        return self.create_filter(response)


    def prompt_create_query(self, query):
        prompt = f"""
            You are a query parsing engine for a local search assistant covering three categories of places: hotels, spas, and restaurants.

            Your job: read the user's natural language query and output ONLY a single JSON object describing what they're looking for. No explanations, no extra text, no markdown fences — just the raw JSON object.

            ## STEP 1 — Identify the category
            Choose exactly one value from:
            Categories = {self.categories}

            Use context clues:
            - Food/dish/cuisine words (e.g. burger, pizza, biryani, naan, coffee) → "restaurant"
            - Massage/treatment/wellness words (e.g. massage, spa, aromatherapy) → "spa"
            - Lodging/stay/room words (e.g. hotel, room, pool, gym, wifi, parking) → "hotel"

            ## STEP 2 — Identify filters
            Choose zero or more from these known filter fields:
            filters = {self.filter}

            Extract values only if explicitly present or clearly implied in the query:
            - "name": a specific place name mentioned by the user (e.g. "Royal Inn")
            - "city": a city name (e.g. "Helsinki", "Espoo", "Vantaa")
            - "region": a region name (e.g. "Uusimaa")
            - "rating": a minimum rating if mentioned (e.g. "rated above 8", "9+ rating") — output as a number
            - "distance": a distance value if mentioned (e.g. "within 5km", "under 3 km") — output as a string in the same unit given, e.g. "5km"

            If a filter is not mentioned, its value must be null.

            ## STEP 3 — Identify requested service(s)
            Based on the category chosen in Step 1, match the user's requested item(s) to the CLOSEST valid entries in that category's service list below. Do not invent services that aren't in the list. If the user's word doesn't exactly match a list entry, map it to the nearest matching entry in that list (e.g. "burger" -> "Burgers", "massage" -> the closest specific massage type mentioned, or omit if too vague to map).

            hotel_services = {self.hotel_services}

            spa_services = {self.spa_services}

            restaurant_services = {self.restaurant_services}

            If the user requests more than one service, include all of them as a list, in the order mentioned.
            If no specific service is mentioned (e.g. "find me a hotel in Helsinki"), output an empty list: []

            ## OUTPUT FORMAT
            Always return exactly this JSON structure, with no other keys, no commentary, and no markdown code fences around it:

            {{
            "Category": "<hotel | spa | restaurant>",
            "name": <string or null>,
            "city": <string or null>,
            "region": <string or null>,
            "rating": <number or null>,
            "distance": <string or null>,
            "service": [<matched service strings>]
            }}

            ## RULES
            - Output valid JSON only — no prose before or after it.
            - Use null (not "None", not empty string) for any filter that wasn't mentioned.
            - "service" is always a list, even for a single service, even if empty.
            - Match service names exactly as spelled in the category's service list (correct casing, correct plural/singular form).
            - If the query mixes signals from multiple categories, pick the category that best matches the primary intent of the query.
            - Never fabricate a city, region, or name that wasn't stated or strongly implied by the user.

            ## EXAMPLES

            Query: "find me a burger place in 5km in helsinki"
            Output:
            {{"Category": "restaurant", "name": null, "city": "Helsinki", "region": null, "rating": null, "distance": "5km", "service": ["Burgers"]}}

            Query: "I want burger and naan near vantaa"
            Output:
            {{"Category": "restaurant", "name": null, "city": "Vantaa", "region": null, "rating": null, "distance": null, "service": ["Burgers", "Naan"]}}

            Query: "any good spa in espoo with hot stone and couples massage, rated above 9"
            Output:
            {{"Category": "spa", "name": null, "city": "Espoo", "region": null, "rating": 9, "distance": null, "service": ["Hot Stone Massage", "Couples Massage"]}}

            Query: "hotel with a pool and free wifi within 10km of helsinki"
            Output:
            {{"Category": "hotel", "name": null, "city": "Helsinki", "region": null, "rating": null, "distance": "10km", "service": ["Swimming Pool", "Free WiFi"]}}

            Query:
            {query}
            
            Output:
            {{"Category": "hotel", "name": "Royal Inn", "city": null, "region": null, "rating": null, "distance": null, "service": []}}
        """
        return prompt

# obj = queryCreator() 
# key = ""
# model = ChatGroq(groq_api_key=key, model_name="openai/gpt-oss-safeguard-20b")

# print(obj.create_query(model, "somewhere for haleem and kheer close to me, under 6km, in helsinki"))


# test_cases = [ 
#     # --- RESTAURANT: multiple services ---
#     {"query": "I want burger and naan near vantaa",
#      "expected": {"category": "restaurant", "Name": None, "City": "Vantaa", "Region": None, "rating": None, "distance": None, "service": ["Burgers", "Naan"]}},
#     {"query": "any place with biryani, chicken karahi and naan in helsinki rated above 8.5",
#      "expected": {"category": "restaurant", "Name": None, "City": "Helsinki", "Region": None, "rating": 8.5, "distance": None, "service": ["Biryani", "Chicken Karahi", "Naan"]}},
#     {"query": "show me somewhere with shawarma and french fries within 4km in espoo",
#      "expected": {"category": "restaurant", "Name": None, "City": "Espoo", "Region": None, "rating": None, "distance": "4km", "service": ["Shawarma", "French Fries"]}},

#     # --- RESTAURANT: no service / just category+filters ---
#     {"query": "find a restaurant in vantaa rated above 9",
#      "expected": {"category": "restaurant", "Name": None, "City": "Vantaa", "Region": None, "rating": 9, "distance": None, "service": []}},
#     {"query": "best restaurant in uusimaa region",
#      "expected": {"category": "restaurant", "Name": None, "City": None, "Region": "Uusimaa", "rating": None, "distance": None, "service": []}},

#     # --- Filter-only / edge cases ---
#     {"query": "hotels in uusimaa",
#      "expected": {"category": "hotel", "Name": None, "City": None, "Region": "Uusimaa", "rating": None, "distance": None, "service": []}},
#     {"query": "best rated restaurant within 3km",
#      "expected": {"category": "restaurant", "Name": None, "City": None, "Region": None, "rating": None, "distance": "3km", "service": []}},
#     {"query": "spa rated above 9.5 in helsinki",
#      "expected": {"category": "spa", "Name": None, "City": "Helsinki", "Region": None, "rating": 9.5, "distance": None, "service": []}},

#     # --- Ambiguous / tricky wording ---
#     {"query": "somewhere for haleem and kheer close to me, under 6km, in helsinki",
#      "expected": {"category": "restaurant", "Name": None, "City": "Helsinki", "Region": None, "rating": None, "distance": "6km", "service": ["Haleem", "Kheer"]}},
#     {"query": "deep tissue and sports massage place in espoo, at least 8 rating",
#      "expected": {"category": "spa", "Name": None, "City": "Espoo", "Region": None, "rating": 8, "distance": None, "service": ["Deep Tissue Massage", "Sports Massage"]}},
#     {"query": "budget hotel with breakfast in vantaa under 5km",
#      "expected": {"category": "hotel", "Name": None, "City": "Vantaa", "Region": None, "rating": None, "distance": "5km", "service": ["Breakfast"]}},

# ]


# for i in test_cases:
#     q = i["query"]
#     print("\n",q)

#     print(obj.create_query(model, q))

#     print("expected: ", i["expected"])