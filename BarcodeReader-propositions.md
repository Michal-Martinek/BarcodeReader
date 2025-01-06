# Čtečka čárových kódů - _BarcodeReader_
*Michal Martínek*

## **Popis zadání projektu**
Cílem projektu je vytvořit aplikaci, která dokáže detekovat a číst čárové kódy z obrázků.  
Hlavní výzvou projektu je navrhnout algoritmus pro identifikaci klíčových informací v obrázku a současně aplikovat principy zpracování obrazu v praxi.

Projekt se zaměřuje na ověření teoretických vizí v reálném prostředí a získání praktických zkušeností s návrhem a implementací algoritmů pro analýzu obrazu.

## Hlavní funkcionalita zahrnuje:
* Zachycení obrazu z kamery nebo načtení z uloženého souboru.
* Algoritmické zpracování obrazu:
	* Předzpracování obrazu:
		* zajištění správné velikosti vstupního obrázku
		* konverze na odstíny šedi
		* binarizace s ohledem na převažující odstíny obrázku
	* Lokalizaci čárového kódu pomocí detekce hran a anylýzy startovních značek
	* Dekódování čárového kódu - čar a mezer
	* Ověření správného načtení kódu
* Zobrazení výsledků v grafickém rozhraní aplikace


### **Vytyčené cíle**  
1. Navrhnout a implementovat základní algoritmus pro detekci a dekódování čárových kódů.  
1. Aplikace bude schopna číst čárové kódy za ideálních podmínek, tedy při dobrém osvětlení a přímém pohledu na kód.  
2. Zajistit, aby aplikace dokázala rozpoznat a interpretovat běžné formáty čárových kódů (např. EAN-13).  
1. Vytvořit aplikaci spustitelnou v OS Windows, s grafickým rozhraním
<!-- 3. Zabalení aplikace do spustitelného souboru (formát .exe) pro snadné použití na operačním systému Windows.  
4. Naučit se pracovat s Python knihovnami pro zpracování obrazu a pochopit základy algoritmů pro detekci a analýzu čárových kódů.   -->

Hlavním cílem projektu je realizace funkčního prototypu,  
který demonstruje základní principy fungování čtečky čárových kódů,  
spíše než dosažení vysoké rychlosti nebo spolehlivosti.

### **Vývojové prostředí**  
Projekt bude vyvíjen v Pythonu s využitím knihoven OpenCV pro zpracování obrazu a NumPy pro numerické operace.  
Aplikace bude určena pro operační systém Windows a výsledná implementace bude nejspíše zabalena do spustitelného souboru.  

