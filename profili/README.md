Profili di configurazione pronti per la visualizzazione di oggetti semi-statici.

Questi file non vengono caricati automaticamente dal programma: il runtime continua a leggere `Config.yaml`.

Contenuto:
- `semi_static_stable.yaml`: preset conservativo e leggibile per target quasi fermi.
- `semi_static_mvdr.yaml`: preset con maggiore separazione angolare tramite MVDR.
- `semi_static_frozen_bg.yaml`: preset utile se puoi avviare il sistema con scena vuota e poi introdurre il target.
- `beamforming_verify.yaml`: preset pulito per verificare ordine canali, simmetria angolare e picco frontale.

Linee guida rapide:
- Se vuoi una vista stabile e semplice da interpretare, parti da `semi_static_stable.yaml`.
- Se vuoi piu risoluzione angolare, prova `semi_static_mvdr.yaml`.
- Se il clutter statico e forte e puoi inizializzare a scena vuota, prova `semi_static_frozen_bg.yaml`.
- Se vuoi validare il beamforming, usa `beamforming_verify.yaml` con un singolo riflettore forte e `NORM OFF`.

Nota pratica:
- Per confrontare i frame in modo coerente, in UI conviene usare `NORM OFF`.
