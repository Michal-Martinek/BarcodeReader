# Čtečka čárových kódů - _BarcodeReader_
*Michal Martínek*

## **Popis zadání projektu**
Cílem projektu je vytvořit aplikaci, která dokáže detekovat a číst čárové kódy z obrázků.  
Hlavní výzvou projektu je navrhnout algoritmus pro identifikaci klíčových informací v obrázku a současně aplikovat principy zpracování obrazu v praxi.  

Projekt se zaměřuje na ověření teoretických vizí v reálném prostředí a získání praktických zkušeností s návrhem a implementací algoritmů pro analýzu obrazu.  

## Hlavní funkcionalita zahrnuje:
* Zachycení obrazu z kamery nebo načtení z uloženého souboru, použití náhodného vstupu z databáze.
* Algoritmické zpracování obrazu:
	* Předzpracování obrazu:
		* konverze na odstíny šedi
		* binarizace s ohledem na převažující odstíny obrázku
	* Lokalizace čárového kódu pomocí detekce hran a anylýzy startovních značek
	* Dekódování čárového kódu - čar a mezer
		- s ohledem na původní odstíny binarizovaných pixelů
	* Ověření správného načtení kódu (_kontrolní součet_)
* Zobrazení v grafickém rozhraní aplikace:
	* vstupního obrázku
	* detekovaného čárového kódu
	* jednotlivých mezikroků rozpoznávacího procesu
	* čtecích čar použitých při detekci
		- včetně jejich detailní analýzy v dialogovém okně
		- možnost zobrazit / skrýt, měnit jejich vzájemnou vzdálenost a sledovat dopad na detekci


### **Vytyčené cíle**  
> Hlavním cílem projektu je realizace funkčního prototypu,  
> který demonstruje základní principy fungování čtečky čárových kódů,  
> spíše než dosažení vysoké rychlosti nebo spolehlivosti.

1. Navrhnout a implementovat základní algoritmus pro detekci a dekódování čárových kódů.  
1. Aplikace bude schopna číst čárové kódy za ideálních podmínek, tedy při dobrém osvětlení a přímém pohledu na kód.  
2. Zajistit, aby aplikace dokázala rozpoznat a interpretovat běžné formáty čárových kódů - **EAN-13**  
1. Vytvořit aplikaci spustitelnou v OS Windows, s grafickým rozhraním
3. Zabalení aplikace do spustitelného souboru (formát .exe) pro snadné použití na operačním systému Windows.  


### **Vývojové prostředí**  
Projekt bude vyvíjen v **Pythonu** s využitím knihoven **OpenCV** pro zpracování obrazu a **NumPy** pro numerické operace.  
Grafické rozhraní bude využívat knihovnu **PyQt6**.  
Aplikace bude určena pro operační systém Windows a výsledná implementace bude zabalena do spustitelného souboru exe.  

## Závěr
Byl realizován funkční prototyp s moderním uživatelským rozhraním, který s jistou spolehlivostí čte čárové kódy standartu EAN-13 a nabízí vhled a povrchní analýzu užitého algoritmu.  

Dle vytyčených cílů byly autorem promyšleny a následně v praxi ověřeny a implementovány algoritmy zpracování obrazu.  
Tudíž nejde o profesionální přístup k řešení problémů tohoto typu, ale pouze o pokus jak dosáhnout cíle postupným obměnováním užitého postupu, dle získaných praktických zkušeností a zjištěných nedostatků.  

### Hlavní nedostatky programu v tuto chvíli:
- správné rozlišení bílé a černé (binarizace vstupu)
	- způsobuje záměnu pixelů, což zamezuje správné funkci dalšího zpracování
		- např. se nenajde značka START/STOP, dva pruhy se slijí do jednoho, záměna čtených číslic
	- to je většinou způsobeno nekvalitním vstupním obrázkem (nízké rozlišení / kontrast, rozmazání, šum na úrovni pixelů, špatné osvětlení)
	- pro minimalizaci tohoto problému bylo podniknuto:
		- adaptivní thresholding - bereme v úvahu jas okolí pixelu
		- výpočet šířky pruhů bere u úvahu jas krajních pixelů
			- _spočítá se kde v prostoru mezi dvěma pixely nejspíše dochází k přechodu bezi barvami -> neceločíselná šířka pruhů_

### Možná další vylepšení
- zlepšení uživatelského komfortu
	- zobrazení náhledu a informací o produktu který byl načten čárovým kódem

- zlepšení binarizace vstupu
	- vyladění parametrů thresholdingu
	- využít širší okolí čtecí čáry (ne pouze 1-pixel širokou)
		- to by omezilo vliv náhodého šumu na záměnu barev
	- předzpracování obrazu pro odstranění šumu
