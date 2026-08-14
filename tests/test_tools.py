import unittest

from agent import enrich_company, render_evidence, search_directory


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

    def test_renderer_keeps_people_titles_paired_with_companies(self):
        text = render_evidence("Healthcare people", [search_directory("Healthcare")])
        self.assertIn("Lena Kovac — Board Member at Verdant Health", text)
        self.assertIn("Lena Kovac — CFO at Aurora BioLabs", text)
