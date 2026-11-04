"""Name pools per nationality. Faker where a locale yields Latin-script names, otherwise short
curated lists of common romanised names. Pools are sampled with numpy, so generation scales
to millions of rows without per-row Faker calls."""

from __future__ import annotations

import numpy as np
from faker import Faker

FAKER_LOCALE = {
    "Germany": "de_DE", "Austria": "de_AT", "Czech Republic": "cs_CZ", "Türkiye": "tr_TR",
    "India": "en_IN", "Italy": "it_IT", "Spain": "es_ES", "Poland": "pl_PL", "Brazil": "pt_BR",
    "Colombia": "es_CO", "Mexico": "es_MX", "Vietnam": "vi_VN", "Indonesia": "id_ID",
    "Philippines": "en_PH", "Ecuador": "es_ES",
}

# (male first names, female first names, last names)
CURATED: dict[str, tuple[list[str], list[str], list[str]]] = {
    "China": (
        ["Wei", "Hao", "Jun", "Lei", "Yang", "Chen", "Bo", "Tao", "Jie", "Ming"],
        ["Li", "Xin", "Yan", "Jing", "Mei", "Ying", "Hui", "Xiaoyu", "Lan", "Fang"],
        ["Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou", "Xu", "Sun"],
    ),
    "Iran": (
        ["Ali", "Reza", "Mohammad", "Hossein", "Amir", "Mehdi", "Saeed", "Arash", "Kian", "Babak"],
        ["Fatemeh", "Zahra", "Maryam", "Sara", "Niloufar", "Parisa", "Shirin", "Leila", "Mina", "Roya"],
        ["Hosseini", "Ahmadi", "Mohammadi", "Rezaei", "Karimi", "Moradi", "Jafari", "Rahimi", "Kazemi", "Ghasemi"],
    ),
    "Pakistan": (
        ["Ahmed", "Bilal", "Usman", "Hamza", "Hassan", "Faisal", "Imran", "Zain", "Saad", "Omar"],
        ["Ayesha", "Fatima", "Hira", "Sana", "Maryam", "Zainab", "Amna", "Mahnoor", "Iqra", "Noor"],
        ["Khan", "Malik", "Hussain", "Qureshi", "Chaudhry", "Butt", "Shah", "Raza", "Iqbal", "Siddiqui"],
    ),
    "Syria": (
        ["Mohammad", "Ahmad", "Khaled", "Omar", "Yousef", "Rami", "Hasan", "Samer", "Fadi", "Tarek"],
        ["Rasha", "Lina", "Hiba", "Nour", "Dima", "Reem", "Sara", "Rama", "Yara", "Hala"],
        ["Al-Ahmad", "Haddad", "Khoury", "Mansour", "Darwish", "Saleh", "Hamdan", "Nasser", "Aziz", "Hassan"],
    ),
    "Egypt": (
        ["Mohamed", "Ahmed", "Mahmoud", "Mostafa", "Karim", "Youssef", "Amr", "Tamer", "Hany", "Sherif"],
        ["Nour", "Mariam", "Salma", "Yasmin", "Dina", "Aya", "Heba", "Rana", "Shaimaa", "Mona"],
        ["Ibrahim", "Abdelrahman", "Hassan", "Farouk", "Mostafa", "Ali", "Fathy", "Saad", "Kamel", "Shawky"],
    ),
    "Morocco": (
        ["Youssef", "Mehdi", "Hamza", "Anas", "Ayoub", "Omar", "Rachid", "Karim", "Imad", "Adil"],
        ["Salma", "Imane", "Khadija", "Hajar", "Meryem", "Sanae", "Nadia", "Houda", "Asmae", "Ghita"],
        ["El Amrani", "Benali", "Alaoui", "Idrissi", "Bennani", "Tazi", "Berrada", "Chraibi", "Fassi", "Ouali"],
    ),
    "Jordan": (
        ["Yazan", "Laith", "Odai", "Hamza", "Mohammad", "Zaid", "Anas", "Faris", "Qais", "Tamer"],
        ["Raneem", "Tala", "Dana", "Leen", "Sara", "Haya", "Rawan", "Shahd", "Bayan", "Lujain"],
        ["Al-Masri", "Haddad", "Khalil", "Abu-Zaid", "Odeh", "Qasem", "Nasser", "Shami", "Majali", "Rawashdeh"],
    ),
    "Tunisia": (
        ["Aymen", "Skander", "Wassim", "Bilel", "Mehdi", "Oussama", "Fares", "Hamza", "Amine", "Nidhal"],
        ["Rania", "Ines", "Mariem", "Amira", "Sarra", "Yosra", "Emna", "Asma", "Olfa", "Hela"],
        ["Ben Salah", "Trabelsi", "Gharbi", "Jaziri", "Bouazizi", "Hamdi", "Mejri", "Chaabane", "Sassi", "Ayari"],
    ),
    "Nigeria": (
        ["Chinedu", "Emeka", "Oluwaseun", "Tunde", "Ibrahim", "Chukwuemeka", "Ifeanyi", "Babatunde", "Segun", "Uche"],
        ["Ngozi", "Chioma", "Adaeze", "Funmilayo", "Aisha", "Blessing", "Temitope", "Amaka", "Yetunde", "Folake"],
        ["Okafor", "Adeyemi", "Okonkwo", "Balogun", "Eze", "Nwosu", "Bello", "Adebayo", "Ogunleye", "Obi"],
    ),
    "Ghana": (
        ["Kwame", "Kofi", "Kwabena", "Yaw", "Kojo", "Kwaku", "Nana", "Kwesi", "Fiifi", "Ebo"],
        ["Ama", "Akosua", "Abena", "Efua", "Adwoa", "Afia", "Esi", "Yaa", "Akua", "Ekua"],
        ["Mensah", "Boateng", "Owusu", "Asante", "Osei", "Appiah", "Agyeman", "Darko", "Ansah", "Amoah"],
    ),
    "Cameroon": (
        ["Jean", "Paul", "Serge", "Arnaud", "Herve", "Boris", "Franck", "Brice", "Yannick", "Cedric"],
        ["Marie", "Carine", "Sandrine", "Linda", "Estelle", "Diane", "Laure", "Nadege", "Prisca", "Vanessa"],
        ["Ngono", "Mbarga", "Tchoumi", "Fotso", "Kamga", "Nkoulou", "Essomba", "Tagne", "Njoya", "Atangana"],
    ),
    "Kenya": (
        ["Brian", "Kevin", "Dennis", "Collins", "Kipchoge", "Otieno", "Mwangi", "Kamau", "Omondi", "Wanjiku"],
        ["Faith", "Mercy", "Wanjiru", "Akinyi", "Njeri", "Achieng", "Grace", "Nyambura", "Atieno", "Wambui"],
        ["Mwangi", "Otieno", "Kamau", "Kariuki", "Odhiambo", "Kiprop", "Njoroge", "Wafula", "Chege", "Mutua"],
    ),
    "Ethiopia": (
        ["Abebe", "Dawit", "Yonas", "Tesfaye", "Bereket", "Henok", "Mulugeta", "Samuel", "Kaleb", "Biruk"],
        ["Hanna", "Selam", "Meron", "Tsion", "Bethlehem", "Saba", "Liya", "Rahel", "Helen", "Eden"],
        ["Tadesse", "Bekele", "Haile", "Gebre", "Alemu", "Tesfaye", "Mekonnen", "Girma", "Wolde", "Desta"],
    ),
    "Bangladesh": (
        ["Rahim", "Tanvir", "Sakib", "Arif", "Fahim", "Nayeem", "Rakib", "Shuvo", "Imtiaz", "Mahmud"],
        ["Nusrat", "Farzana", "Tasnim", "Sadia", "Sumaiya", "Mim", "Tania", "Nazia", "Lamia", "Jannat"],
        ["Rahman", "Hossain", "Islam", "Ahmed", "Chowdhury", "Uddin", "Akter", "Sarkar", "Miah", "Haque"],
    ),
    "Nepal": (
        ["Bikash", "Suman", "Prakash", "Rajesh", "Anish", "Sandeep", "Bibek", "Nabin", "Sujan", "Roshan"],
        ["Sita", "Anjali", "Pooja", "Srijana", "Sabina", "Asmita", "Pratiksha", "Sushma", "Bina", "Rojina"],
        ["Shrestha", "Thapa", "Gurung", "Tamang", "Adhikari", "Karki", "Rai", "Magar", "Poudel", "Bhandari"],
    ),
    "Sri Lanka": (
        ["Kasun", "Nuwan", "Chamara", "Tharindu", "Dilshan", "Lahiru", "Sahan", "Isuru", "Pasindu", "Ruwan"],
        ["Nadeesha", "Dilini", "Sachini", "Hiruni", "Tharushi", "Kavindya", "Chathurika", "Anjali", "Nethmi", "Ishara"],
        ["Perera", "Fernando", "Silva", "Jayasinghe", "Bandara", "Wickramasinghe", "Dissanayake", "Herath", "Rajapaksa", "Gunawardena"],
    ),
    "Ukraine": (
        ["Oleksandr", "Andrii", "Dmytro", "Serhii", "Maksym", "Ivan", "Bohdan", "Yurii", "Taras", "Mykola"],
        ["Olena", "Iryna", "Kateryna", "Anastasiia", "Yuliia", "Oksana", "Daryna", "Sofiia", "Viktoriia", "Mariia"],
        ["Shevchenko", "Kovalenko", "Bondarenko", "Tkachenko", "Kravchenko", "Melnyk", "Boiko", "Moroz", "Lysenko", "Savchenko"],
    ),
    "Greece": (
        ["Georgios", "Dimitrios", "Konstantinos", "Ioannis", "Nikolaos", "Panagiotis", "Christos", "Vasileios", "Athanasios", "Evangelos"],
        ["Maria", "Eleni", "Aikaterini", "Vasiliki", "Sofia", "Angeliki", "Georgia", "Dimitra", "Konstantina", "Paraskevi"],
        ["Papadopoulos", "Papadakis", "Georgiou", "Nikolaidis", "Konstantinidis", "Pappas", "Oikonomou", "Vlachos", "Makris", "Dimitriou"],
    ),
}


class NamePools:
    def __init__(self, nationalities: list[str], seed: int, pool_size: int = 400):
        self.pools: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for nat in nationalities:
            if nat in CURATED:
                m, f, last = CURATED[nat]
                self.pools[nat] = (np.array(m), np.array(f), np.array(last))
                continue
            fk = Faker(FAKER_LOCALE.get(nat, "en_US"))
            fk.seed_instance(seed)
            self.pools[nat] = (
                np.array(sorted({fk.first_name_male() for _ in range(pool_size)})),
                np.array(sorted({fk.first_name_female() for _ in range(pool_size)})),
                np.array(sorted({fk.last_name() for _ in range(pool_size)})),
            )

    def sample(self, rng: np.random.Generator, nationality: np.ndarray, gender: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = np.empty(len(nationality), dtype=object)
        last = np.empty(len(nationality), dtype=object)
        for nat in np.unique(nationality):
            idx = np.flatnonzero(nationality == nat)
            m, f, ln = self.pools[nat]
            g = gender[idx]
            # "diverse" draws from either pool.
            use_f = (g == "female") | ((g == "diverse") & (rng.random(len(idx)) < 0.5))
            first[idx] = np.where(use_f, rng.choice(f, len(idx)), rng.choice(m, len(idx)))
            last[idx] = rng.choice(ln, len(idx))
        return first, last
