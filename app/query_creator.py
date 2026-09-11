import json

class queryCreator:
    def __init__(self):
        self.categories =  ["spa", "hotel", "restaurant"]
        self.filter = ["Name" , "City", "Region" , "rating" , "distance" ] 
        self.hotel_services = ['Free WiFi', 'Swimming Pool', 'Gym', 'Restaurant', 'Parking', 'Room Service', 'Airport Shuttle', 'Spa', 'Breakfast', 'Garden', 'Private Dining']
        self.spa_services = ['Thai Massage', 'Swedish Massage', 'Deep Tissue Massage', 'Hot Stone Massage', 'Aromatherapy', 'Couples Massage', 'Foot Massage', 'Head Massage', 'Sports Massage', 'Facial Massage']
        self.restaurant_services = ['Biryani', 'Chicken Karahi', 'Chicken Tikka', 'Naan', 'Raita', 'Burgers', 'Pizza', 'French Fries', 'Chicken Wings', 'Pasta', 'Mutton Karahi', 'Seekh Kebab', 'Kheer', 'Sandwiches', 'Shawarma', 'Gulab Jamun', 'Chocolate Cake', 'Coffee', 'Haleem']

    def create_filter(self, response):
             
        data = json.loads(response)
    
        query_filter = {}

        field_mapping = {
            "category": "Category",
            "City": "city",
            "Region": "region",
            "rating": "rating",
            "distance": "distance"
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
                query_filter["service"] = {"$in": services}


        return query_filter

    def prompt_create_query(self):
        prompt = f"""
            You are a query parsing engine for a local search assistant covering three categories of places: hotels, spas, and restaurants.

            Your job: read the user's natural language query and output ONLY a single JSON object describing what they're looking for. No explanations, no extra text, no markdown fences — just the raw JSON object.

            ## STEP 1 — Identify the category
            Choose exactly one value from:
            categories = {self.categories}

            Use context clues:
            - Food/dish/cuisine words (e.g. burger, pizza, biryani, naan, coffee) → "restaurant"
            - Massage/treatment/wellness words (e.g. massage, spa, aromatherapy) → "spa"
            - Lodging/stay/room words (e.g. hotel, room, pool, gym, wifi, parking) → "hotel"

            ## STEP 2 — Identify filters
            Choose zero or more from these known filter fields:
            filters = {self.filter}

            Extract values only if explicitly present or clearly implied in the query:
            - "Name": a specific place name mentioned by the user (e.g. "Royal Inn")
            - "City": a city name (e.g. "Helsinki", "Espoo", "Vantaa")
            - "Region": a region name (e.g. "Uusimaa")
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
            "category": "<hotel | spa | restaurant>",
            "Name": <string or null>,
            "City": <string or null>,
            "Region": <string or null>,
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
            - Never fabricate a City, Region, or Name that wasn't stated or strongly implied by the user.

            ## EXAMPLES

            Query: "find me a burger place in 5km in helsinki"
            Output:
            {{"category": "restaurant", "Name": null, "City": "Helsinki", "Region": null, "rating": null, "distance": "5km", "service": ["Burgers"]}}

            Query: "I want burger and naan near vantaa"
            Output:
            {{"category": "restaurant", "Name": null, "City": "Vantaa", "Region": null, "rating": null, "distance": null, "service": ["Burgers", "Naan"]}}

            Query: "any good spa in espoo with hot stone and couples massage, rated above 9"
            Output:
            {{"category": "spa", "Name": null, "City": "Espoo", "Region": null, "rating": 9, "distance": null, "service": ["Hot Stone Massage", "Couples Massage"]}}

            Query: "hotel with a pool and free wifi within 10km of helsinki"
            Output:
            {{"category": "hotel", "Name": null, "City": "Helsinki", "Region": null, "rating": null, "distance": "10km", "service": ["Swimming Pool", "Free WiFi"]}}

            Query: "show me Royal Inn"
            Output:
            {{"category": "hotel", "Name": "Royal Inn", "City": null, "Region": null, "rating": null, "distance": null, "service": []}}
        """
        return prompt

# obj = queryCreator()

# print(obj.prompt_create_query())