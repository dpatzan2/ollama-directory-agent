import unittest

from agent import enrich_company, search_directory


class DirectoryToolTests(unittest.TestCase):
    def test_search_connects_healthcare_companies_to_people(self):
        result = search_directory("people at Healthcare companies")
        names = {person["full_name"] for person in result["people"]}
        self.assertEqual(names, {"Grace Okafor", "Lena Kovac", "Daniel Osei"})

    def test_enrichment_returns_every_candidate(self):
        result = enrich_company("Helios Data")
        self.assertEqual(len(result["candidates"]), 2)

    def test_search_prefers_an_exact_company_name(self):
        result = search_directory("Helios Data")
        self.assertEqual([company["company_name"] for company in result["companies"]], ["Helios Data"])

    def test_enrichment_reports_no_match(self):
        self.assertEqual(enrich_company("Sable Security")["candidates"], [])
