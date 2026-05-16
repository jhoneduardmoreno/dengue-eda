---
fecha: 2026-04-24
titulo: Seguimiento de avance del proyecto de dengue con Carlos
director: Carlos
participantes:
  - Jhon Edwar Moreno Diaz
  - Hernán
  - Andrés
  - Amaya (Amalia)
  - Carlos (director)
tema: Feedback sobre informe de avance — modelado, partición de datos, métricas, despliegue
---

# Reunión de seguimiento — Carlos (2026-04-24)

## Decisiones accionables (TL;DR)

Feedback clave de Carlos para nuestro grupo y aplicable al proyecto:

1. **Enfocar el modelo en una región, no en todo el país.** Colombia es muy heterogénea (climas, culturas, dinámicas de almacenamiento de agua). Un modelo nacional no puede generalizar bien sin meter variables socioeconómicas, lo cual está fuera del alcance del proyecto. Recomendación: elegir 1 a 3 municipios o departamentos y entrenar **un modelo independiente por cada uno**.
2. **Criterios para elegir esos lugares** (cualquiera es válido, pero hay que justificarlo):
   - Lugares con **más casos absolutos** (impacto en salud pública agregada).
   - Lugares con **mayor incidencia por habitante** (casos / población) — captura las regiones más vulnerables que a menudo quedan invisibilizadas.
   - Lugares de **interés particular** (curiosidad, contexto local, disponibilidad hospitalaria).
3. **Usar toda la data disponible, no sólo años representativos sueltos.** Lo que tenemos hoy (2010, 2016, 2019, 2022, 2024) deja huecos enormes; conviene completar los años intermedios.
4. **Partición train/test cronológica, no aleatoria.** En series de tiempo no se hace split al azar: el test debe ser el último ~20% de los años (p. ej. entrenar 2007–2018 y testear 2019–2024). El test del 2024 solo es un caso atípico (año con exceso) y por sí solo no basta.
5. **Métricas a priorizar: recall y precision.** Accuracy es engañoso con clases desbalanceadas.
   - **Recall** = qué tanto de los excesos reales capturo (alarmismo bajo si es bajo).
   - **Precision** = cuando el modelo dice "hay exceso", qué tanta confianza puedo tener.
   - En nuestra matriz actual: precision ≈ 0.43, recall ≈ 0.87 → modelo "alarmista" poco confiable cuando predice exceso.
6. **No forzar deep learning con dataset pequeño.** A escala semana × municipio×s pocos años tenemos ~780 puntos por municipio: insuficiente para LSTM/Transformers/TFT con `hidden_size=64+`. Si se intenta, bajar `hidden_size` a 2–8. Una **regresión logística regularizada o XGBoost** suele ser suficiente y robusta.
7. **Despliegue = prototipo.** No es necesario integrar APIs de Google Earth Engine en tiempo real. Basta con que la interfaz consuma la tabla de Excel ya generada y haga la predicción para el año de test.
8. **Datos 2025: opcional.** Si no salen a tiempo en SIVIGILA, no afecta la entrega. Si salen, úsenlos como segundo test sin reentrenar.

## Anexos / referencias mencionadas

- **Marco Geoestadístico Nacional (DANE)** — capas geográficas de hospitales por municipio, útil si se cruza disponibilidad hospitalaria.
- **bbycroft.net** — visualización interactiva de arquitecturas de transformers (nano-GPT, GPT-2, GPT-3).
- **SIVIGILA** — fuente de datos epidemiológicos de dengue (aún sin datos 2025 al momento de la reunión).
- Modelos discutidos por otros grupos: Poisson, LightGBM, LSTM, TFT (Temporal Fusion Transformer).

---

## Discusión por grupo

### Grupo 1 — Jhon Edwar, Hernán, Ruby, Danilo

**Lo que presentaron:**
- Datos de todas las regiones del país.
- Años de entrenamiento seleccionados de forma representativa (≈ 2007, 2008, 2010, 2012).
- Año de prueba: **2024**.
- Modelos: Regresión Logística, Random Forest, XGBoost.
- Métrica priorizada: **Recall** (por tratarse de datos desbalanceados, con ponderación de clase negativa).
- Mejor modelo según recall: **Regresión Logística**.

**Preguntas planteadas a Carlos:**
- ¿Aplicar técnicas de balanceo en los datos?
- ¿Probar modelos temporales?
- ¿Enfocarse en regiones específicas?
- ¿Conviene usar todos los años en lugar de años representativos?

**Feedback de Carlos:**
- **Matriz de confusión actual:** la precisión real para "exceso" es ~43% y el recall ~87%. Es un modelo *alarmista*: detecta la mayoría de los excesos reales, pero también genera muchas falsas alarmas. Esto se debe principalmente a que se está mezclando toda la heterogeneidad espacial del país.
- **Recomendación principal:** **enfocarse en una única región** (un departamento, un municipio, o un conjunto pequeño de municipios) y/o entrenar modelos independientes por región. Esto evita tener que modelar explícitamente la heterogeneidad espacial.
- **Usar toda la serie de años, no solo años "representativos".**
- **Modelos:** los actuales (regresión logística regularizada, XGBoost) son **suficientes y meritorios**. Deep learning (RNN, Transformers) opcional como extra.
- **Partición de datos en series de tiempo:** **no se hace al azar**. Se debe coger el último ~20% como test. Ejemplo: entrenar 2007–2018 y testear 2019–2024.
- **Métricas a reportar:**
  - **Accuracy:** útil pero sesgada por desbalanceo. Reportarla pero no confiar plenamente.
  - **Precision** (TP / (TP + FP)): "cuando el modelo dice que hay exceso, ¿qué tan confiable es?"
  - **Recall** (TP / (TP + FN)): "de todos los excesos reales, ¿cuántos detecta el modelo?"
  - Las **dos clave** son **Precision y Recall** sobre el test.

**Pregunta de Hernán — ¿cómo elegir la región?**
- **Opción A:** lugares con **mayor número absoluto de casos** (Ibagué, Girardot, etc.). Útil para mitigar el mayor número de casos.
- **Opción B:** lugares con **mayor incidencia por habitante** (casos / población). Útil para enfocarse en comunidades vulnerables y de mayor riesgo (Amazonas, Vichada, etc.). Evita el sesgo de "solo donde hay más gente hay más casos".
- **Opción C:** lugares de **interés particular** (por una motivación específica, comunidad olvidada, etc.) — también válido.
- **Recurso recomendado:** *Marco Geoestadístico Nacional* del **DANE** — contiene datos georreferenciados de hospitales por municipio, útil para cruzar con disponibilidad hospitalaria.

**Pregunta sobre Transformers (curiosidad de Jhon):**
- Los Transformers "clásicos" ya están siendo reemplazados/evolucionados en arquitecturas más complejas (apilamiento de capas de atención, múltiples cabezas, atención cruzada, enmascarada, etc.). La filosofía sigue vigente, pero las arquitecturas modernas (GPT-3+, multimodales) son mucho más complejas. Referencia visual: `bbycroft.net` (visualizador de LLMs).

### Grupo de Andrés

**Lo que presentaron / preguntaron:**
- Estado de datos 2025: aún no publicados en SIVIGILA.
- Avance en integraciones con Google Earth Engine y el Marco Geoestadístico Nacional.
- Modelos en uso: **Poisson** y **LightGBM** (LightGBM con mejores resultados; posible uso solo para explicabilidad).
- Exploración de **LSTM** y **TFT** (Temporal Fusion Transformer) en PyTorch — sin superar a LightGBM.
- Tres municipios trabajados: Ibagué, Girardot, Espinal.

**Feedback de Carlos:**
- **Datos 2025:** no esperarlos. Usar hasta 2024 es lo esperado por el proyecto. Si llegan, usarlos como segundo test.
- **Despliegue:** mantenerlo simple como **prototipo**. La entrada debe ser la **tabla Excel ya procesada**, no APIs en vivo de Google Earth Engine. Lo de la API es un extra, no requisito.
- **Deep learning con pocos datos — explicación clave:**
  - 2007–2022 entrenamiento ≈ 15 años × 52 semanas ≈ **780 datos**.
  - Un LSTM con hidden_size = 64 ya tiene >200 parámetros → **overfitting casi garantizado** con tan pocos datos.
  - Recomendación: bajar `hidden_size` a **2, 4 u 8** si insisten en deep learning.
  - El grupo confirma: con hidden_size 250–500, train mejora pero test se degrada (overfit, agravado por 2024 atípico).
  - El deep learning brillaría si tuvieran ~30 años a escala diaria, o muchos municipios × semanas (~702.000 datos).
- **Modelos independientes por municipio:**
  - Idealmente, **un modelo entrenado por cada municipio** (mismos hiperparámetros pueden cambiar entre municipios). Cada municipio = unidad independiente.
  - Andrés aclara que entrenan con todos los municipios e infieren sobre tres — Carlos sugiere idealmente independizarlos, pero **si ya está avanzado el pipeline, no echar atrás**.
- **Métrica agregada entre municipios:** si se trabajan como un solo bloque, **sumar** los errores en lugar de promediarlos (es más coherente cuando se tratan como una sola unidad).
- **Prioridad:** terminar primero un entregable funcional (aunque sea borrador), luego pulir.

### Grupo de Amaya

**Lo que presentaron:**
- Dos modelos comparativos:
  1. Modelo **nacional** (~900 municipios).
  2. Modelo con **3 departamentos** representativos.
- Incluyeron ciclo de vida del mosquito y período de incubación de la enfermedad como variables.
- Resultados: **el modelo de 3 departamentos da mejores métricas** (Recall, Precision, F1, ROC) que el nacional.

**Feedback de Carlos:**
- Es un fenómeno típico de **escala espacial**: a escala nacional, las variables relevantes se promedian y se pierden las dinámicas locales (clima, cultura, comportamiento). A escala departamental, ciertas variables (sociales, climáticas locales) cobran más peso.
- **Recomendación:** quedarse con el modelo a escala departamental (el que está dando mejor desempeño) para construir el despliegue.
- Sería interesante explorar también escala nacional si queda tiempo, pero priorizar el de mejor performance.

---

## Acciones / próximos pasos para el grupo de Jhon

1. **Re-entrenar enfocándose en una región específica** (a definir como grupo según criterio: mayor número absoluto de casos, mayor incidencia por habitante, o interés particular).
2. **Usar toda la serie temporal disponible** (no solo años representativos).
3. **Reajustar la partición train/test:** último ~20% del rango temporal como test (p. ej. 2019–2024 test, anteriores como train).
4. **Reportar Precision y Recall** sobre el test como métricas principales (además de Accuracy y F1).
5. **Mantener regresión logística y XGBoost como modelos base**; deep learning queda opcional y, si se intenta, con `hidden_size` muy pequeño (2–8).
6. **Considerar cruzar con disponibilidad hospitalaria** usando el Marco Geoestadístico Nacional del DANE.
7. **Construir un dashboard prototipo** con entrada por archivo Excel ya procesado; conexiones por API son opcionales.

---

## Transcripción completa

**Carlos:** Creo que no hay todavía una fecha superconfirmada, pero sí supondría que es el segundo semestre del dos mil veintiséis. Bueno, no, genial. Excelente. Espero verlos a todos y todas ahí. ¿Qué vas a preferir si no hacer la sesión, porque, pues aún no conocía como a nadie, para el otro sí esperaría estar con ustedes allá. Listo. Bueno, chicos, entonces voy a, igual, a grabar la sesión, como para que pueda, digamos, igual ahí demás poderla subir y que ustedes la puedan revisar o los otros compañeros que de pronto no se puedan conectar hoy. Y puedo ponerla aquí en la nube. Listo. Perfecto. Y el objetivo de esta sesión es como poder hacer un seguimiento, de pronto, sí, ya con base en lo que entregaron en este informe de avance y durante esta semana, si tienen alguna duda, alguna pregunta, si quieren, de pronto, mostrarme algo que hayan hecho como grupo, a ver si está bien, mal, tener un poco de feedback y demás, sería como la gracia de de vernos aquí. Poder aprovechar y, de pronto, revisar esas preguntas que ya les quedan faltando, como para darles un último empujón, y poder, digamos así, aprovechar la sesión de la siguiente semana y las otras que tendremos, para que ya salga el producto final y les vaya superbién en lo que es este proyecto de Grau. Entonces, no sé, grupo quiera empezar, ¿cierto? De pronto, ahí sí, con todo el el el respeto cariño, sí priorizaría un poquito más a grupos que de pronto no hayamos hablado, ¿sí? ¿Cierto? O sea, con el grupo de Andrés, con el grupo de Amaya, que hemos podido revisar cositas, Los dejaría como de segundos, pero quisiera saber si, además de estos dos grupos, alguno de los demás tienen alguna duda, quieren comentar algo, quieren que revisemos algo inmediatamente. Igual, es importante que tengan presente que, así yo estoy atendiendo a otro grupo, muchas veces las dudas que le resuelva ese grupo van a poderle ayudar a los otros grupos para poder terminar de forma en dejar lo que queda faltando. Entonces, quedo atento si algún grupo quiere quiere revisar algo, ¿Tiene alguna pregunta, alguna duda? Podéis levantar la manito, pueden hablar sin pena,

**Jhon:** Carlos, ¿cómo estás?

**Carlos:** ¿Qué tal, John?

**Jhon:** Buenas tardes. Por aquí la cámara,

**Carlos:** Buenas tardes. Listo, yo. ¿Qué más, como va todo?

**Jhon:** Muy bien, brother. ¿Tú qué tal?

**Carlos:** Bien, bien, todo en orden, todo en orden aquí, ya cerrando semana.

**Jhon:** Excelente. Carlos, nosotros en mi grupo aquí están donde estaba Hernán, Ruby, Danilo y demás. Utilizamos

**Carlos:** Ok.

**Jhon:** como un recap de lo que hicimos,

**Carlos:** Ok.

**Jhon:** y me gustaría que que nos dieras como tu feedback y qué otras opciones podemos explorar, ¿sí?

**Carlos:** Perfecto.

**Jhon:** Nosotros utilizamos cierto ciertos años representativos, digamos, aplicamos, utilizamos data de de todas las regiones,

**Carlos:** Ajá.

**Jhon:** ahí ahí ahí tú no tú nos dirías si es mejor como enfocarse en una región en específica,

**Carlos:** Ok.

**Jhon:** Y aplicamos ciertos modelos. En este caso, aplicamos regresión logística y XG

**Carlos:** Ok.

**Jhon:** el de de

**Carlos:** ¿Quién sabe ser?

**Jhon:** el de y digamos, compramos el

**Carlos:** Y

**Jhon:** el recall, que, en este caso, pues, como trabajando con data de balanceada y demás, es como el el indicativo de cuál sería el mejor modelo y nos dio nos dio

**Carlos:** Bravo.

**Jhon:** regresión logística y regresión logística. Obviamente, digamos, al trabajar con estos modelos, nosotros utilizamos ese ese parámetro que nos nos nos permite como que más la clase negativa, ¿sí? En cada uno de estos modelos.

**Carlos:** Sí.

**Jhon:** Y no tomamos todos los años, tomamos solamente cuatro años, si no tengo el dos mil siete, dos mil ocho, dos mil diez, dos mil doce, o no me acuerdo, no me acuerdo exactamente los años. Pero tomamos tomamos como unos años representativos En ese caso, ¿qué nos recomiendas? ¿Nos recomiendas o sea, como como próximo paso a seguir. ¿Nos recomiendas...? No sé si si te comparto, ya tienes acceso al al informe que nosotros enviamos, te lo comparto, para que tengas más contexto.

**Carlos:** Perfecto. Yo aquí tengo acceso, yo, pero si me lo quieres mostrar desde ahí, también puede ser válido, pero en lo único que no sé es ¿ahí vos puedes compartir?

**Jhon:** Déjame déjame ver. No, ahí, bueno, ahí te envié la solicitud.

**Carlos:** Ah, listo, perfecto, ya ya te lo probé.

**Jhon:** Voilà, ici. Yo comparto y ahí nos nos

**Carlos:** Listo.

**Jhon:** tu feedback. Y otras opciones podemos experimentar, ¿sí?, a a a bacán de talachillo. Y te comparto de una. Me avisa cuando lo estes viendo, por

**Carlos:** Listo, Joan, ahí está cargando. Listo, ahí ya lo estoy bien.

**Jhon:** Listo.

**Carlos:** Ah, supericuito, muy bien.

**Jhon:** Listo. Como te comentaba, los integrantes

**Carlos:** Es

**Jhon:** digamos, pasemos a lo clave rápidamente. Ahí me dice si quiere ver, quieres pausar en algo en específico.

**Carlos:** ¿Listo?

**Jhon:** Nosotros utilizamos estos años para el entrenamiento. Utilizamos el dos mil veinticuatro como conjunto de prueba, ¿sí?

**Carlos:** Ok.

**Jhon:** ¿Qué qué modelos utilizamos? Básicamente, la creación logística, random forest,

**Carlos:** Xeo.

**Jhon:** ¿Sí? ¿Qué nos dieron nos dieron la el resultado con con respecto al al recall, no dio la regresión logística

**Carlos:** Mira,

**Jhon:** digamos, los mejores resultados.

**Carlos:** Perfecto, sí.

**Jhon:** Entonces, mi pregunta va dirigido sobre qué otras opciones promover explorar, por ejemplo, nosotros aplicamos ese esa ese ese parámetro y ponderamiento, pero digamos, durante el entrenamiento de los modelos, no aplicamos como algunas técnicas en la data. Sé si nos recomiendas aplicar alguna técnica en la data. Estamos trabajando con con datos de balanceados, No sé si nos recomiendes irnos también como probar algunos otros modelos, no sé, estos modelos más temporales, sí, nos nos enfocamos solamente en algunas regiones, no como todas las regiones del el país, y qué otra cosa podríamos tener en cuenta, por decirlo así.

**Carlos:** Listo, John, perfecto, excelente. Entonces, seguimos. Hay un par de cositas importantes que podamos, digamos, trabajar y revisar, que no son digamos que lo chévere es que ya, con el trabajo que han hecho, John, cualquier modificación que se haga ya están los códigos, ya está, digamos, todo el trabajo, ya está toda esa parte, digamos que realmente relativamente fácil hacer cambios, ¿no? Digamos,

**Jhon:** Si, si, si.

**Carlos:** sí, lo primero que te quería preguntar yo, este en este caso, cuando haces esta matriz de confusión como para mirar, digamos, la regresión logística que fuera elegida, con base en este a u c, tenía la pregunta de, ¿esto que me muestras acá es con base en los datos de entrenamiento o de test? Con los datos que dices de esos como tres años que entrenaron o con los datos del dos mil que fue el test que utilizaron.

**Jhon:** No. No, esto con el en base al al mil veinticuatro.

**Carlos:** Listo, combate al dos mil veinticuatro, perfecto. Entonces, ahí me da también otra preguntica, y es que ustedes, según vi, creo, cruzaron como años como dos mil siete, luego usaron como el dos mil no sé, digamos, doce, pero los años entre esos dos no los utilizaron, ¿cierto?

**Jhon:** No, esa otra pregunta. Porque no te estamos pensando ahorita utilizar ya toda la

**Carlos:** Ajá.

**Jhon:** ¿sí? Entonces, ¿tú lo recomiendas?

**Carlos:** Claro,

**Jhon:** Aplicarlo a toda la data, digamos,

**Carlos:** Y ahí entra otra cosita importante, John, respecto a lo primero que me preguntaste. Cuando nosotros revisamos estos dos casos, ¿cierto?, en Colombia, algo, digamos, que sucede, que es muy importante tener en cuenta, es que nuestro país, pues nosotros acá en Colombia siempre decimos que somos un superdiverso, ¿sí? Y ojo que, digamos, esa diversidad nuestra no solo viene por el hecho de que Los Andes llega a calmas y son colombianos y se los divide, en estas tres cordilleras, ¿cierto?, a que nos demarca una corregiones superdistintas entre ellas, que tenemos altitudinales super ¿no? O sea, tenemos desde Orinoquía, desde Sabana, desde Mazórica, desde manglares costeros hasta hasta nevados y, pues, todos los ecosistemas que hay a lo largo de ese gradiente, No solo tenemos esa diversidad de animales, también tenemos una diversidad enorme de climas, ¿sí? Por ejemplo, hacia la parte de la Caribe, los climas normalmente son un verano muy marcado a inicio de año, y luego un pico de lluvias, pero es algo que se conoce como monomodal. Mientras que en la zona de las montañas tenemos patrones climáticos de lluvias, en las cuales hay dos picos de lluvias. Y, en parte, es como la Amazonía tenemos un verano, entre comillas, que realmente no es que sea un verano como en el Caribe, que llueve cero, que es un verano donde igual llueve un montón, pero llueve un poquito menos. ¿Sí? Hay toda una diversidad de patrones climáticos Y, además de eso, John, en nuestro país tenemos la característica de que esa diversidad biológica, y climática también da origen a una diversidad cultural enorme. ¿Cierto? Uno va sale de Bogotá y llega al Tolima, y pasa por lugares con distintas culturas, distinta música, hasta distintos marcas de aguardiente. O sea, es una vaina así increíble, ¿no? Entonces, digamos que todo esto hace que, al momento en el cual yo uso una base de datos, para todo Colombia, sea muy difícil poder generalizar. Porque, en este caso que estamos hablando del dengue, obviamente, hay un factor climático que va a afectar las de los mosquitos de una manera clara, pero también hay un factor cultural importante asociado a por ejemplo, las dinámicas de almacenamiento de agua. ¿Sí? Acá, en Bogotá, es muy raro ver que las personas mantengan tanques de agua. ¿Sí? Igual, por ejemplo, en zonas como en el Llano, o en Leticia, muy raro ver eso. Pero en regiones como Ibaqué, en regiones de bosque seco tropical, es más común ver personas que almacenen agua en distintos lugares, para lavar, sea para, digamos, bañarse y demás, para el momento en el cual hay escasez de agua, puedan tener esos reservorios de agua y eso las dinámicas del dengue. ¿Qué sucede aquí? Ya meternos con todas esas variables socioeconómicas, guion, es posible, ¿cierto?, es posible, pero realmente yo, como director de estos proyectos, no lo recomendaría, pues, porque acá obviamente, está la parte investigativa, innovativa del proyecto, pero está la parte de que, pues, ustedes tienen que graduarse, ¿no? Porque son mejor intentar abarcar algo como un poco más delimitado. Mi consejo, yo, sincero, respecto a esa pregunta, y también lo digo muy claro por esto que se ve aquí en tu matriz de confusión. ¿Sí? Mira que, por ejemplo, cuando nosotros hablamos para la parte que nosotros que estamos en una epidemia, que tenemos un exceso de casos, ¿cierto? Hay más casos de los esperados, Mira que las actividades de las predicciones se reparten igual entre las reales que no hay exceso y que hay excesos. Realmente, digamos, si uno se pone a pensar, en este caso, ¿sí?, en mirar qué tan robusta es la estimación que hace nuestro modelo, va a ser muy poco robusta, que, prácticamente, aquí, cuando yo hablo de que tiene un acceso, hay un cincuenta por ciento de probabilidad, incluso un poquitico menos, de que realmente le pega que un es eso. Claro, y yo estoy apostaría mucho a que eso es por el hecho de que no estamos integrando todas esas dinámicas sociales, comportamentales que hay en los distintos lugares del país. Mi recomendación ahí, John, es enfóquense en un lugar. ¿Cierto? Si quieren, enfóquense, de pronto, en todos los departamentos de algún lugar que sea de su interés. Y el hecho de que sea de su interés puede ser que tenga la mayoría de casos, pero iba a primar en Tolima, va primar el Valle Del Cauca, ¿sí? Pueden simplemente elegir otro lugar que también es muy útil, eso porque a veces con este tipo de problemas en epidemiología, normalmente hay un sesgo de que las herramientas desarrollan para los lugares donde más pasan casos, pero en los lugares no igual pasan casos, que igual hay gente afectada y poblaciones afectadas quedan olvidado. ¿Sí? Realmente, mi recomendación ahí

**Jhon:** Yep.

**Carlos:** es que los modelos que están usando están chéveres, ¿sí? Obviamente, uno los puede complejizar, John, uno puede pensar en meter por deep learning, incluso redes neuronales recurrentes, ¿sí? O incluso meter transformers. Pero muchas veces, para este tipo de datos que tenemos, una regresión logística, con las modificaciones que ustedes le hacen, que es la regularización para evitar que haga un overseat, o hasta incluso un XGhost, unos árboles de decisión que son más sencillos, tienen menos parámetros que una red neuronal profunda. Van a ser suficientes y van a poder hacer un trabajo bastante bueno. ¿Sí? Entonces, mi recomendación sería esa. Respecto a los modelos que están usando, se me hacen señales para tener esa base. ¿Listo? Y se me hacen buenos y, digamos, meritorios de que ustedes tengan una buena nota en este proyecto.

**Jhon:** Listo.

**Carlos:** ¿Listo? Ya, si les queda el tiempo, chévere que le metan algo más complejo, que le metan una red neuronal si les quieren meter redes neuronales, o algo de deep learning, ¿cierto?, o también si les quieren meter, de pronto, algún otro tipo de regreso. Ya está, digamos, en ustedes. Pero realmente con ese baseline de modelos que eligieron, a mí se me hace que está bien. ¿Listo? Porque también hay que tener en cuenta que hay un trabajo de coger los datos satelitales, organizar toda una base de datos, depurarle y todo eso, es un trabajo de management que es bastante importante y valioso. Entonces, por eso yo esos modelos los veo bien. Pero los modelos no están sirviendo bien, John, ¿cierto? Así tengo un buen a u c aquí, en esta partecita, que yo predico que hay un realmente esa predicción está siendo no tan buena, ¿sí? No tengo buena confianza en esa predicción, Hay un sesenta, cincuenta por ciento de probabilidad de que, si digo, cae el ceso no lo vaya, pues, eso ya hace que sea muy poco robusta la predicción, John, Está porque están usando todos los municipios y, pues, realmente, la inteligencia artificial no tiene las variables necesarias para poder hacer esa partición espacial. ¿Listo? Y ahí mira que ustedes ahí se están enfrentando dos cosas. A una, parte de la heterogeneidad de los casos espaciales, y la heterogeneidad temporal, realmente, es mejor enfocarse solo en lo temporal, eligiendo un único sitio. Que esa sería mi recomendación, John, ¿listo? Que se enfoque en un solo sitio, pueden elegir un departamento, pueden elegir un municipio, pueden elegir dos departamentos, pueden elegir tres municipios, ¿cierto? Hay grupos que están haciendo tres departamentos independientes, cogen los datos de un departamento y entrenan el modelo. E, independientemente para otro departamento, entrenan el mismo modelo, pero, pues, con otro ajuste de datos, ¿cierto?, con otro set de parámetros que encuentre el modelo tras hacer el ajuste. Y eso les va a permitir tener muchísimas digamos, métricas y muchísima mejor capacidad. Creo que eso es lo principal, John. Como que, digamos, puedan hacer eso elegido en una región de interés ¿listo?, no en todo el país, sino en una región de interés, o en varias regiones de interés, pero que haga modelos independientes en cada región. Para que ustedes no le estén metiendo el modelo o no tengan que enfrentarse a meter de la característica que les separe esos distintos comportamientos espaciales, sino que asuman que k unidad espacial va a ser distinta y el modelo va a ajustarse de manera diferente.

**Jhon:** Buenísimo. Otra pregunta, Carlos.

**Carlos:** ¿Listo, John?

**Jhon:** Nosotros, por ejemplo, para t s utilizamos dos mil veinticuatro.

**Carlos:** Sí.

**Jhon:** Te voy a comentar. Pero el dos mil veinticuatro hubo un exceso. Si es si era utilizar ese año como O sea, un exceso mayor a lo a los años anteriores.

**Carlos:** Claro. Sí, John, algo importante es que también pueden utilizar ahí digamos, normalmente en este tipo de situaciones, algo que se bastante es, al momento de hacer mi partición de datos, en usar el train y el test, ¿cierto? Digamos que esa es toda mi serie de tiempo de caso, ¿cierto? Esta grafiquita, al caso, son los casos, respecto al tiempo. Bien. ¿Sí? Cuando se hace la partición, normalmente, uno está acostumbrado a coger el test de, digamos, lo va a poner de color verdecito, y empezar a elegir regiones de esta, de forma al azar, para tener el texto. ¿Sí? Esa forma, en series de tiempo en predicciones, no es tan usada y no es válida. ¿Por qué? Porque, normalmente, nosotros lo que vamos a hacer es querer predecir de aquí en adelante, ¿cierto? Por esto, muchas veces, lo que se hace es coger, digamos, el último veinte por ciento de los datos como el test. ¿Qué quiere decir esto? Que, digamos, si ustedes se entregan con datos de, digamos, del dos mil seis, hasta el dos mil dieciocho, digamos que usan del dos mil diecinueve, hasta el dos mil veinticuatro para test, por ejemplo, ¿sí? De esa manera, ustedes van a poder evaluar si su modelo ajustado con datos del pasado, que es todo lo que tenemos de ese línea hacia atrás, que va a ser el train. ¿Sí?, con esos datos, ustedes realmente van a poder estar prediciendo el futuro de forma correcta y están generalizando lo que se quiere entender, que es en esta parte del test. ¿Cierto? Acordémonos que cuando usamos este tipo de inteligencias artificiales aquí, el ideal lo que nosotros quisiéramos como ingenieros o como, digamos, expertos en inteligencia artificial, el mundo ideal es que esta inteligencia artificial capture todos los mecanismos que están pasando que hagan que Que la gente viaje, que el mosquito nazca, que los huevos se etcétera, etcétera, etcétera. Que todo eso se vea reflejado, de alguna manera, en nuestra algoritmo de inteligencia artificial con los ajustes, ¿sí? Va a ser bastante difícil hacer esto, pero si yo con los datos del trade, que es el pasado, soy capaz de predecir un test que va ser el futuro? Que no se pueda Que estamos rebiendos. Nuestro modelo extrajo información, mecánica y hasta incluso así no lo sepamos a extraer información que puede estar relacionado con una causalidad de distintas variables y demás, que me permita hacer esa me permite realmente modelar qué es lo que está pasando para que se den dos casos, ¿listo? Por eso, este tipo de partición de datos, es la que más se utilice.

**Jhon:** Una última pregunta, calor, que tampoco quiero como marcar todo el tiempo para la misma compañía.

**Carlos:** No, tranquilo, yo en cero lío.

**Jhon:** ¿Qué qué métrica nos recomiendas? ¿Sí? Si elegimos una métrica adecuada, evaluar esto?

**Carlos:** La métrica que recomendaría acá, acordémonos que aprovechando que tienes la matriz de confusión ahí, ¿cierto?, acordémonos que nosotros podemos tener cuatro métricas iniciales cuando trabajamos con exceso de casos o no exceso de casos, con variables binarios, ¿cierto? Podemos tener la primer métrica, que es el accuracy, en el cual yo lo que estoy mirando es, a cuántos le pegué bien, ¿cierto? Sobre el total. ¿Listo? Esta métrica es muy general y, pues, puede tener ciertos problemas con desbalanceo de datos entre las categorías. ¿Sí? Que después por eso le apuro así, es útil reportarlo y calcularlo pero no es tan útil al momento de confiar. Entonces, digamos, a este apuro sí le vamos a hacer como una carita medio feliz, ¿cierto? O sea, que es bueno reportarlo y calcularlo, pues uno tampoco se demora mucho calculándolo, pero digamos que nos puede permitir tener una primera aproximación, pero realmente no es como la más útil porque está sesgado a el balance de los datos. El siguiente que nosotros podemos tener, la siguiente métrica que nosotros podemos tener, ¿cierto?, es la que conocemos como la precisión. ¿Listo? Entonces, vamos a ponerla aquí. Y acordémonos que en la precisión lo que a mí me interesa es mirar cuántos categoricé que tuvieron exceso y realmente tuvieron exceso de casos, ¿listo? Vean que acá no me están interesando los que categoricé como que no tenían exceso. Esos no los estoy utilizando. Pero lo que me va a importar de ahí, realmente en esta precisión, ¿listo?, son los verdaderos positivos que son estos de acá, ¿cierto? Y también nos van a importar aquí los falsos positivos. ¿Cuáles son los falsos positivos? Los falsos positivos son en los cuales yo dije que había Era mucho más de lo que que realmente, ¿cierto? Cuando hablamos de el falso positivo, nosotros hablamos de que me un momentico para no irla a embarrar. ¿Cierto? Cuando tenemos un falso positivo, lo que nosotros decimos es que no tuvimos exceso de casos, pero yo predije que había un exceso de casos. Mira que con la precisión nosotros estamos mirando. Dado que mi modelo dijo que hay un exceso, ¿qué tan cierto es ese exceso? Cuando mi modelo dijo, alístese, preocúpese porque se vino a exceso de casos, Realmente, ¿qué tanto le puede confiar al modelo? Esa es la precisión. Y mira, por ejemplo, John, en tu caso en particular, esa precisión, si la calculamos es de mil ciento ochenta, sobre mil quinientos sesenta y dos más mil ciento ochenta. ¿Listo? Y ese valor nos da pongo la calculadora aquí del lado, se Y ese valor nos da mil ciento ochenta sobre mil quinientos sesenta y dos, más mil ciento ochenta. Mira que, en este caso, la precisión es de el cuarenta y tres por ciento, es cero punto cuarenta y tres. ¿Qué quiere decir? Que cuando este modelo que ustedes calibraron dice, va a haber un exceso de casos yo solo puedo tener un nivel de confianza del cuarenta y tres por ciento. Si es que ese ya nos está mostrando que hay algún ahí, que realmente cuando el modelo dice, oiga, alístese, póngase trucha, no puedo confiar en él. Entonces, mira que la precisión me va dar una mejor información en este caso. ¿Listo? La siguiente métrica que tenemos, que va ser interesante, en toda la represión le voy poner una carita feliz. La siguiente métrica que tenemos, que es bastante interesante, es la que conocemos como el record. Y el recall es parecido a lo que es la precisión, pero en el recall, yo cojo ¿cuántos fueron positivos si yo dije que eran positivos? ¿Cierto? Pero dividido entre el total fueron positivos. Mira, aquí, tu en tu caso, no es sorry, calls, ¿cierto?, es mil ciento ochenta sobre ciento sesenta y cinco, más mil ciento ochenta. Y esto nos da un valor de perdió la calculadora. Pero mira que, en este caso, el recall va a ser mucho mejor. No. Ciento ochenta sobre ciento treinta y cinco más mil ciento ochenta. En ese caso, el record es de cero punto ochenta y siete. ¿Qué quiere decir? Que tu algoritmo es capaz ¿cierto?, de calcular el ochenta y siete por ciento de los casos en los cuales hubo exceso de casos. Entonces, mira que, a diferencia de la precisión, ¿qué pasa? Si hay un exceso de casos, tu modelo va a ser muy bueno capturándolo, porque tiene un del ochenta y siete por ciento. Pero si tu modelo dice que hay exceso de casos, puede que no haya exceso de casos. Entonces, es un modelo que uno lo podría llamar como muy alarmista. ¿Cierto? Es un modelo que puede generar pánico. Al generar pánico, va a capturar la mayoría de veces que ese pánico sea real pero muchas de las veces que genera pánico no eran necesarios, dice. Entonces, por eso, estas dos métricas, ¿cierto?, van a ser las métricas que yo más te recomiendo en este caso. ¿Listo?, lo que es el recall y lo que es la precisión. Mejorando esas dos métricas, vas a tener la capacidad de tener un modelo que si tiene esas dos métricas buenas, es un modelo que cuando pase alguna embarrada, o sea, cuando hay un exceso de casos, cuando tenemos que prepararnos para atender muchos casos de aire, va a ser muy probable que el modelo lo capture y nos prepare bien, pero además de eso, cuando el modelo diga que hay que prepararse, podemos confiar bastante en por eso las dos principales que yo te recomiendo es el recall y el precision, principalmente en los datos de test. Con eso tú vas a saber, cuando hago mi modelo ajustado con ciertos datos, y lo corro en datos del futuro que nunca se han visto, puedo mirar qué tan bueno es el módem, y de esa manera vas a tener la capacidad de poder decir, mejorémoslo o de pronto vestir jazz, dejémoslo listo. ¿Listo, John?

**Jhon:** No, buenísimo, muchas gracias, muchas

**Carlos:** Listo, pero

**Jhon:** Carla.

**Carlos:** John, todo bien. A vos ahí por la participación, que es importante. Una una cosita antes de cambiar al siguiente grupo. Profe, por Claro.

**Hernán:** Yo recuerdo, sí, en una de sesiones anteriores, si no estoy mal, hace hace quince, efectivamente, uno de los grupos mencionó que había tomado unas zonas específicas, iba a Girardot, ese esa zona. Tomando lo que nos acabas de responder, ¿qué criterios en la últimas usamos tomar esa zona? O sea, o o qué zonas va a ser donde haya mucho mucha prevalencia del o lo miramos por altura sobre el nivel del mar, o lo miramos por población total de la región? ¿Qué qué nos recomiendas ahí enfocar ese ese tema y esa selección?

**Carlos:** Hernán, esa pregunta es, supercrucial y superválida y superimportante, porque ya depende de qué enfoque les quieran dar al proyecto, ¿cierto? Ok. Ok. Grupo que eligió esos lugares es porque buscó esos lugares lo que son, digamos, Girardot, lo que es Ivalid y demás, son los lugares con el mayor número de casos ¿cierto? Ok, ok. Pasa en este caso. Que tú estás enfocando tu herramienta a eso, ¿sí? Y ojo que este grupo, lo que hizo fue obtener, digamos, tres sitios diferentes Pensemos lo que es que desea alquilar dos, pensemos lo que sea ibaguet, pensemos lo que sea, no sé, el espinal. Y lo que hacen es, cada uno, hacer todo su entrenamiento y calibración del modelo de inteligencia artificial, de manera independiente, ¿claro? ¿Cómo para qué? No tenga que buscar alguna variable que me lo sepa y usar una inteligencia artificial general, sino que cada uno tiene su AI independiente, lo cual está superbién, ¿verdad?, porque es una manera en la cual podemos evitar estas complejidades de heterogeneidades que les he estado hablando ahorita. Claro, Hernán. Claro. ¿Qué pasa? Cuando uno hace esto, que es que uno pensaría en liberación de epidemiología, nosotros estamos teniendo un tipo de en el cual estamos enfocando todos los recursos a las zonas donde ocurren más gastos, pero el hecho de que hayan casos depende de que hayan, ¿sí? ¿Qué quiere decir esto? Que nosotros vamos a estar enfocando nuestros recursos y toda esta herramienta a las zonas donde que está la enfermedad y donde hay más gente expuesta. Y ahí estamos cometiendo, digamos, no es un error, porque no está mal, pero estamos cometiendo un sesgo muy importante, y es que no donde tengo más casos reportados va a ser las comunidades que más municipales. ¿Cierto? Pensémoslo, por ejemplo, en uno de los temas, pues, que yo más trabajo en morleduras de serpientes, el departamento que más reporta casos es y, con base en eso, todos los esfuerzos deberían irse a Antioquia. Claro, de los cinco mil casos, más o menos, que ocurren aquí, un poco más del diez por ciento, como seiscientos y péguele, ocurre en Antioquia. Si yo elijo Antioquia, digo, me voy a enfocar en Antioquia, pues está bien, es el lugar donde ocurren más casos, y sí logro, digamos, no sé, prevenir o mejorar la situación, hay seiscientos casos que logro reducir y que, pues, voy a poder, efectivamente, mitigar el número grande de casos en el país. ¿Sí? ¿Cuál es el sesgo ahí, Hernán? Que realmente las zonas donde hay mayor riesgo, o sea, si yo no miro, cuántos casos netos ocurren, sino miro cuántos casos ocurren, dado que la población es de este tamaño, ¿cierto? Y esa epidemiología lo puedo ver muy similar, y le pongo comillas, pues, por no es como tan bueno decirlo así, pero es para que se entienda más claro. Yo lo que calculo es la probabilidad de que si meto a una persona en ese lugar, lo vuelva una culera, ¿sí? Cuando yo hago esa corrijo mis casos por la población en la cual ocurrieron esos casos, ese me prende la alerta, Amazonía, que se me prende la alerta, bichada, toda la falta de Amazonía, tráfico de lobby no alquilar el país. Antes no se me prendió, pues, porque son lugares donde no hay mucha gente. Obviamente, por pura probabilidad, no van a haber mucho casos. Pero cuando yo miro qué tan en riesgo están las personas de ahí, el riesgo de perder una mordedura en esas regiones puede ser tres veces mayor al de Antioquia. Pero como no hay tanta gente, no terminan generando casos efectivos y por eso a veces quedan siendo, digamos, sobrevistos o totalmente, digamos, ignorados por la salud pública. Hernán? Entonces, ya para responder tu pregunta como mirando ese contexto, pueden elegir los lugares donde ocurran más casos, está superbién. Pueden elegir los lugares donde, si cogen el número de casos promedio, anuales y los dividen en la población del lugar, van a obtener una variable que se conoce como incidencia por cada habitante. ¿Cuántos casos ocurren por habitante por ahí? Ahí van a poder saber cuáles son los lugares que están en mayor riesgo. ¿Listo? Esa es otra manera de elegirlo o la tercera es el lugar que les interese. Por x o y motivo. También está bien, si ustedes es de pura curiosidad, mirar cómo suceden o cómo se pueden predecir esos casos en Fusagasugá, que van a haber pocos casos porque es alto, pero hay ciertos casos, pues ese tipo de monitoreo es importante, porque, eventualmente, la población de foods a gas u gas se va a ver beneficiada por esto. ¿Listo? ¿No? Entonces, realmente, si ustedes quieren hacerlo como enfocado a mitigar la mayoría de casos, ¿cierto?, la la opción es elegir los lugares que más reporten caso. Si ustedes quieren Súper, claro. Poner el foco en las regiones más vulnerables y de mayor riesgo, entonces, elijan donde ocurran más casos por habitante. Y, si quiere otra razón, pues está también bien. ¿Cierto? Si de repente quieren elegir el municipio de Pitalito, en el ¿por qué un abuelito de allá y se les hizo interesante? También está superbién. ¿Cierto? Porque es una comunidad que, si yo lo miro a nivel general, va a quedar, digamos, olvidada por ese tipo de elección. ¿Listo, Hernán?

**Hernán:** Superclaro, profe. Si llevamos la abstracción, a lo a lo de las serpientes, de alguna manera, tú lo que hiciste fue hacer un pareto y, ah, bueno, es lo más representativo, y por allá me fui.

**Carlos:** Exacto, en ese caso, pero porque mi situación era, listo, yo me quiero enfocar, en las zonas más vulnerables, en las zonas donde hay más riesgo y entender por qué hay más riesgo ahí. Si es que las serpientes son más agresivas, o si es que las personas, por sus actividades cotidianas, están más expuestas. Pero porque es más probable que una persona la muerdan ahí que en otro lugar. Y por eso ahorita yo estoy haciendo todo un trabajo superfuerte de campo en el ¿sí? Pero digamos que todo depende de en qué vamos se quiera enfocar. Si, por otro lado, yo quisiera enfocarme en reducir los casos que se reportan, pues lo mejor es ir anteope. Hay más casos, tengo más probabilidad de que sí hago una intervención efectiva en reduzco la mayoría de casos. Pero no me estoy enfocando en las comunidades vulnerables, ¿listo?

**Hernán:** Profe. Superclaro, ahí nos llevamos una buena idea. De pronto, vamos a ver si si si lo terminamos cruzando con tema de disponibilidad hospitalaria o algo así, que puede ser una

**Carlos:** Eso sería interesante como como como para verlo. Se me ocurre en el Entonces, nos dio mucha luz. Muchas gracias, profe. Hernán, si de quieren ir por ahí, les recomiendo, y anótenlo, el marco geoestadístico Ese marco geoestadístico nacional está publicado por el departamento de estadística de Colombia, el y ahí ustedes información de georreferenciada de los hospitales de todo el país. Con eso, pueden por municipio cuántos hospitales hay y elegir el que tenga la menor cobertura. Súper. Súper. ¿Listo?

**Hernán:** Listo, profe. Muchas gracias.

**Carlos:** Perfecto. Yo creo que ahora sí el siguiente grupo. Trabajen y cualquier cosa nos vemos el otro viernes, si me cuentan cómo les va. Gracias, Listo. Entonces, antes de seguir con, de pronto, con el grupo de Amalia o el grupo de Andrés, ¿hay algún otro grupo que tenga alguna duda que quiera que revisemos algo? Listo, entonces, Amaya, Andrés, ¿tienen alguna duda, quieren comentar algo, quieren revisar algo? Sé si te tengo que seguir con Amari. ¿Empieza ahí de una? Listo.

**Andrés:** Pues nosotros, en en el informe, hicimos como una sección de como de los retos y hallazgos que tenemos pendientes por solucionar Pues, mejor dicho, sí, el que el el principal, yo creo que que ya lo hemos hablado con con usted varias ocasiones, es el tema de los datos. Yo personalmente estoy obsesionado entrando a a a al portal civil a como las semanas, a ver si ya actualizaron los datos del dos mil veinticinco, veo que todavía no. No, eso se va Ok. Entonces, sí, yo yo eso es como un primer punto, yo creo que tal vez un un un tema de alineación era, si de aquí a a lo que del trabajo salen esos datos pues yo creo que ya nos vamos con lo que nosotros entrenamos del dos mil ¿cierto? Porque, pues, sería un retroceso bastante fuerte

**Carlos:** No tiene sentido, Andrés. Realmente, de pronto, lo que podrían hacer, pero ojo, que eso no es lo esperado dentro de los objetivos del proyecto, sería un extra. Que sería superbacano, ¿no?, pero es un extra, ¿no? No es como lo que se espera lo que ustedes hagan, porque al momento que empezamos y al momento que hemos trabajado, toca tener en cuenta la disponibilidad de los datos. Y eso era lo que sucede full en investigación en esto. Uno depende de qué tan disponibles estén los datos, ¿no? Sería superdiferente, e incluso no tendría problemas, digamos, Andrés, su merced o su grupo estuvieran trabajando directamente en el Instituto Nacional de Salud, en el área de análisis de datos, predicciones, salud y clima, etcétera, etcétera, ¿sí? Por más que sean en el instituto, si el si vigilase demora subiendo los datos, también ustedes van a quedar colgados. Entonces, es algo absolutamente normal, Andrés, y por eso digamos que esperamos que usen los datos más actualizados, sino, cuando lo miramos, los que habían disponibles, que es hasta el dos mil veinticuatro, y está superbién con eso. Ahora, si salen los del dos mil veinticinco y están despachados y les queda tiempo de sobra, miren cómo con el modelo que ya calibraron, miren qué tal se la juega haciendo la estimación en dos mil veinticinco, úsenlo como un segundo test. ¿Sí? Que, pues, no es algo que les va a consumir mucho tiempo. Pero realmente ya tienen el modelo, funciona bien con lo que están trabajando y demás, con eso, Andrés, y está superbién. ¿Vale?

**Andrés:** Buenísimo, listos. ¿Listo?

**Carlos:** También tengan en cuenta, pues, que todo el pipeline que ustedes desarrollaron desde la parte de la selección, digamos, de los municipios, hasta la parte de calibrar los modelos, todos los códigos, todo eso, es un trabajo muy importante que si salen los datos del dos mil veinticinco y ustedes quisieran poderlo hacer, pues ya tienen todo eso desarrollado. Y si no son ustedes, si no es de pronto otro instituto, otra entidad, pues ya tiene toda esa base que es casi que todo el trabajo que hay que hacer. ¿Sí? Entonces, por eso yo no me preocuparía tanto porque no usan los datos de dos mil veinticinco. Chévere usarlos del carajo, pero, pues, si no están eso no les va a quitar validez ni robustez ni excelencia al trabajo que ustedes ¿Listo?

**Andrés:** Buenísimo, listo. Ahí como superalineados. Lo otro que que iba a mencionar es acerca de las integraciones, Digamos que en otras sesiones pasadas también habíamos conversado como de tener la mayor de datos integrados en el en el dashboard, como para facilitarle la vida al usuario y y que sea la menor cantidad de trabajo manual en en el cargue de información y todo el cuento, Sí. Pues, lo tenemos como dentro de los retos, vamos a tratar como de de de ahí lo máximo posible. Eso sí, no es menor, porque, pues, bueno, tenemos los datos de de Google Earth Engine, que a su vez dependen de de los datos que comentaba anteriormente del del marco geodesico nacional, Sí. Y y y, pucha, bueno, ahí ahí hay unas cositas, como unos retos por ver si hay tales para poder llegar ahí, a esas a esas bases de datos, si es fácil de Entonces, bueno, esos son como dentro de los pasos que que estaremos explorando, a ver como qué qué alcance dentro de la integración podemos tener de de de las fuentes. De digamos que para efectos del de avance, nos fuimos con los resultados de dos modelos, que fue Poisson y Live GVM, Ok. En particular, LightGBM creemos que está dando unos buenos resultados, Pues, definitivamente, creo que va a quedar descartado, pero más para como a la parte de explicabilidad del modelo. Ok. Que que nos pueda apoyar más como que su rol no sea de de predicción, sino más de de Claro. Y frente a los modelos de de red de deep learning, red neuronales, que sabemos que que ustedes están como también muy interesados en esa parte, pues vamos a tratar de meterle la ficha en estas semanas que quedan a a dos opciones, que es LCTM y TFT, para para ver cómo nos va. Particular, hemos cómo ha avanzado en varias pruebas de de de TFT, que entendemos que es como el modelo como, digamos, como mejor para los datos de la vida real. Sí. Y que podría llegar a tener mejores resultados, Sin embargo, hasta el momento, los resultados que que hemos tenido no han a los del IGBM. Entonces, pues seguimos explorando, viendo hiperparámetros, viendo como que qué qué opciones tenemos para mejorar ese ese modelo, pero ya lo, pues, como para comentarle que que lo hemos ido también trabajando.

**Carlos:** No, excelente. Hay dos cositas importantes, ¿no?, para la parte de la interfaz, para la parte del despliegue. Tengan en cuenta que igual lo que se espera que ustedes entregan, pues, va a ser un prototipo, ¿cierto? Y dentro de ese prototipo, cuando ustedes y también me gustaría, pues, dejarles este mensaje superclaro para, digamos, si van, no sé, si eventualmente se interesan en seguir por la línea, digamos, de la academia, de pronto, luego hacer otra maestría o hasta incluso entrar a un doctorado, O incluso, digamos, en los momentos en los cuales tengan entregables, ¿cierto?, con alguna empresa y demás, Muchas veces uno quisiera, y pues hay muchas por hacer, siempre, casi que todo trabajo que uno haga y que uno entregue, va a quedar porque siempre van a haber más cosas que hacer. Ahorita la tecnología, ahorita las herramientas de investigación, avanzan un ritmo tan rápido que, pues, pues, pucha, ya está ahorita los transformers se están quedando ¿Cierto? O sea, salieron las redes recurrentes duraron un poquito, y llegaron los transformers con el de Application y SellUnit, y chavos recurrentes, o sea, como que quedaron también obsoletos. Y, pues, vean eso tan tan impactante. Pero es normal, es natural es supernormal que ustedes, de pronto, quieran hacer siete mil millones de cosas pero como hay un deadline de tiempo, solo puedan hacer mil millones de cosas. Pero si esa única cosa que hacen es útil, respecto a lo que voy con ese mensaje es, cuando abran el despliegue de la interfaz, ¿cierto?, lo que se busca es que sea también un prototipo, ¿sí? Entonces, digamos, a esa interfaz, con la conectividad de los APIs, que sería lo ideal con Google Air Engine, ¿cierto? O todo ese tipo de situaciones que yo diga, listo, yo le voy a meter en el municipio, Con base en eso, quiero que él calcule con Google Air Engine el centroide de las coordenadas, y con eso me descargue los rasters y haga el procesamiento para obtener la serie de tiempo de las variables climáticas, y con eso luego yo solo le meto a los que ocurrieron y ya. Sería el mundo ideal, pero, pues, muchas veces, digamos, esa conectividad por APIs, pues también entra, digamos, a una parte más de desarrollo y no tanto de inteligencia artificial. Realmente, si hablamos de una empresa vamos a tener a la persona en inteligencia artificial haciendo todas las herramientas, pero vamos a tener a un desarrollador haciendo el y las conectividades con apps, ¿sí? Por eso tengan en cuenta que ese describir es algo muy en términos de prototipa. Y por eso mi recomendación, Andrés, sigue siendo que la entrada que tenga de todas las variables climáticas sea la tabla de Excel que ustedes ya tienen. Y que, de pronto, ustedes solo cesguen a que hagan las predicciones para el dos mil veinticuatro, que como ese año de test. Y que con base a las variables climáticas que ya tienen para ir cargas para el dos mil y que yo le meta los datos que quiero meterle de casos, haga la predicción, ¿listo? Pero yo no me enfocaría tanto en hacer lo de la API, sino en dejar esa primer paso superhecho, porque ya con eso están bien. Realmente, ya lo de la API es un extra, igual que, digamos, lo otro que venía a decir, y es también estos modelos de deep learning. Es muy bacano que los exploren, son modelos muy potentes pero el problema con el deep learning, ¿cierto?, o, bueno, no el problema, sino la limitación, que tiene deep learning, es que, al crear estas redes neuronales tan amplias, ¿cierto? Y en el caso, digamos, de usar transformers, por ejemplo, en el caso de TFT, o en el caso de usar recurrentes como el LSTM, vamos a tener una situación, y es que, por ejemplo, en los transformers, tenemos que ajustar muchos parámetros. ¿Cierto? Todas las capas de multiatención, todas las capas de atención cruzada, todas las capas de atención enmascarada, todas esas capas dentro del transformer tienen, son matrices de pesos que hay que ajustar. Eso quiere decir que vamos a tener un mundo de parámetros enorme. ¿Listo? En LSTM tenemos que ajustar todos los parámetros de la memoria a largo plazo, ¿cierto?, el self memory o el cell, más los parámetros, digamos, de todas las compuertas de activación e inactivación que tiene LSTM en cada una de las sendas unitarias. ¿Qué va ahí con esto? Cuando lo miramos en términos, ya netamente de herramientas matemáticas, de herramientas numéricas, que es lo que se está usando por detrás, Cuando yo tengo tantos parámetros, pero tengo tan pocos datos, pues, realmente, el objetivo del deep learning, acuérdese, que es, con todas mis capas profundas, las características que me van a permitir hacer mi predicción. Sin que yo tenga que hacer el feature engineering, ¿cierto? Deep learning, el que, a partir de esa extracción, me hace el feature engineering. Si no tengo los suficientes datos, no voy a poder aprovechar al máximo el deep learning. Eso es como, digamos, a la parte que voy, ¿cierto? Digamos, en este caso, nosotros tenemos están trabajando escala de semana epidemiológica, Andrés,

**Andrés:** Sí.

**Carlos:** Listo. Y digamos que ustedes, hasta el dos mil, ¿cuál es el año mínimo que están usando?

**Andrés:** Dos mil siete.

**Carlos:** Dos mil siete, listo. Entonces, tienen, en teoría, y vamos a estar dos mil veinte, ¿hasta qué año están usando para entrenar?

**Andrés:** Dos mil, dos mil veintidós, para entrenar, veintitrés para validación y veinticuatro para test.

**Carlos:** Listo. Entonces, digamos, tienen quince años de entrenamiento, que son los que van a usar para todo el descenso al gradiente a ajustar sus redes de deep learning, son quince años, en los cuales tienen cincuenta y dos semanas. Tienen un total de setecientos ochenta datos. ¿Listo? Uno puede tener fácilmente, digamos, si ustedes hacen un LCTM normal, en el cual uno a veces le pone un key insight de sesenta y cuatro, ahí ya están teniendo como fácilmente más de doscientos parámetros que ajustar, más de doscientos pesos que ajustar. Y si le metes muchas más capas o profundas, van a tener muchísimos más parámetros. Con solo setecientos ochenta datos. Ahí es donde estos algoritmos de deep learning con pocos datos quedan cortos con algoritmos mucho más sencillos como el Claro. Ahora, si nosotros tuviéramos datos de una serie de tiempo de qué sé yo, treinta años a escala diaria, cambia el asunto. Ahora, si, por ejemplo, nosotros también le metiéramos lo que hablábamos ahorita con John, de todas las variables de la heterogeneidad espacial, y usted tuviera esos setecientos ochenta datos semanales para los, pongamos en novecientos municipios que reporte en Dengue, ya son setecientos dos mil datos. Ya el deep learning en a poder aprovechar esa cantidad de datos para poder ajustar su parámetros y extraer características de interés que le permitan mejorar su performance respecto a lo otro. ¿Listo, Andrés? Entonces, es por eso que, en estos casos, el deep learning es chévere, se me hace muy bacano, yo les dije que lo viera porque los viaban adelantados. Entonces, por ahí lo voy ver qué tal. Pero lo más común es que suceda esto. ¿Listo? ¿Cómo puedes mejorar? Reduciéndole la complejidad. Métanle, de pronto, en la parte... ¿Ustedes lo están haciendo que con PyTorch?

**Andrés:** Sí, con PyTorch.

**Carlos:** Exacto. Métanle en la parte, digamos, de su n NL STM, ¿cierto?, cuando definen el modelo, metal de hidden size de ocho. O de cuatro, o incluso de dos. ¿Listo? Pues, nosotros estábamos en las David, Sarmiento, sesenta y cuatro, de hidden size.

**Andrés:** Sí, nosotros estábamos probando como un hidden size muy alto, de doscientos cincuenta, incluso quinientos programas. Pero, pues, lógico, lo que está pasando es que se overfitea en particular, para este problema, el overfeed es genera demasiado ruido, porque año con el que se está testeando es totalmente atípico.

**Carlos:** Exacto, totalmente atípico. Exacto. Entonces, creo, me imagino que ustedes en su tren deben tener así métricas de uno cero punto noventa y cinco, o sea, en el trade de pronto sí mejora, respecto al IGBM, pero en el test se va al canal.

**Andrés:** En el test, ahí sí.

**Carlos:** Se está poniendo de memoria eso y no está extrayendo los mecanismos que causan que pase esto. Esa sería mi recomendación para esos para esos algoritmos. ¿Listo, Andrés?

**Andrés:** Listo, no, buenísimo entendido. Pero John ahí tiene

**Carlos:** Y el Y enfóquese, por también, Andrés, en que ya que tienen el despliegue, hágalo de la manera más sencilla. O sea, mi recomendación ahorita, como investigador, como persona que también pasó por el mal, por doctorado, ¿cierto?, es que ahorita enfóquense ya en con lo que tiene, armar el armar el desarrollo y ya tener como ese entregable que sea muy borrador, que sea muy prototipo, que tenga muchas cosas por mejorar. Pero ya es Alli. Luego eso lo que da es pulir. Es mucho mejor que ponerse a pulir solo el primer paso y luego dejar una semana para hacer un desarrollo que no quede tan bueno.

**Andrés:** Sí, no, de acuerdo. Yo creo que ya para efectos de tener un ya un entregable funcional, tal vez no sea el que prediga mejor, ya estamos yo creo que relativamente cerca. Sí. John creo que tiene una pregunta. Un momentico, qué pena Listo, de una. Entonces, cuéntame, Jol. Que suba, por favor.

**Jhon:** No, Carla, esta es más curiosidad, es que tú que ya los Transformers están quedando atrás, ¿qué tecnología la está reemplazando?

**Carlos:** Digamos que ahorita, con todas estas herramientas, de, pues, toda la innovación que ha partido a partir de los LLMs, ¿cierto?, y de toda esta parte de la idea generativa, El transformer típico, ¿cómo lo entendemos?, con su mecanismo de encoder, decoder, con sus mecanismos de autoatención, atención cruzada, que fue lo que fue el boom con los transformers, realmente ya no es que se quedando totalmente obsoletos, sino que ya se ha construido mucho a partir de ellos, cambiando las arquitecturas. Tanto el encoder, que es el que me permite, digamos, coger mis tokens de lo que yo quiero representar como y poderles hacer el embedding y meterlo en ese lenguaje matemático que usa la red neuronal, que siga estando vigente, pero la de cómo el transformer funciona y cómo va el encoder y cómo el decoder, eso vamos dando cambiando. O sea, apilan capas, se mezclan encoders con otros decoders, se mezclan y capas de atención cruzada. Eso hace que el transformer típico ya, digamos, esté quedando lo que les digo. Ya, digamos, el transformer típico que uno va al inicio realmente, si uno quiere hacer un modelo con un transformer de esos, no va a tener la misma capacidad que tiene la modelo largo de lenguaje ahorita, ¿cierto? Nada más si uno se ve, por ejemplo, a Uy, se me olvidó. Vámonos, Vean, hace tres años, estábamos hablando, digamos, por ejemplo, del GPT cuatro. Y en el GFT4 tenemos cosas distintas con la de tecnología multimodal, ¿listo? Y cuando nosotros vemos, que yo me acuerdo que había una página muy bacana para visualizas, Vamos a ver si lo encuentro así chévere, si no, pues Ok, acá tenemos un transformer normal, canto in the airs. No, este no me muestra los distributions que hay que entender. No, les quedo viviendo esa página, una página muy... Ah, bueno, y por dentro es como esta. Ya les voy a mostrar aquí rápido hacia en el general. Así como de curiosidad, y seguimos con los demás grupos. Vean, por ejemplo, aquí está, vean que apenas acá estábamos hablando de los GPTs. ¿Cierto? Y cuando miramos todo lo que tienen estos, digamos, NanoGPT, cosas muy viejas, ¿cierto? Vean que aquí ya tenemos todos estos key, queries, key y values, ¿cierto?, con todos sus pesos. Vean cómo la arquitectura va cambiando de veinticinco ya tenemos capas apiladas de, digamos, esos mecanismos de atención. Creando múltiples baterías de atención, haciendo que todo eso, digamos, cambie y vean cómo las arquitecturas van cambiando. Si nos vamos al GPT tres, que aún estamos en algo viejito, ahorita, hoy en día, ¿cierto?, vean cómo esa página, BBYCrowth, por si la quieren ahí chismosear, Vean como aquí ya la arquitectura es más grande, es muchísimo más compleja, ¿cierto? Y cambia respecto al transformer tradicional, donde el transformer tradicional es este que ustedes están viendo acá. ¿Cierto? En en normalización, atención cruzada de múltiples cabezas, normalización, feed forward y sal. Esta cosita normal, que es la base, vean cómo cambia con múltiples cabezas de atención, con múltiples, digamos, capas de atención, cruzada y demás, lo cual hace que, digamos, uno haga una innovación cambiando esas arquitecturas, más cosas, y ya, digamos, en transformers clásico cambia totalmente. ¿Listo, George? Sigue siendo como la ideología de un transformer, pero ya las arquitecturas son superdistintas y supercomplejas.

**Jhon:** Listo. Gracias, Carlos.

**Carlos:** Está bien, John. Bueno, ¿alguna otra pregunta, chicos y chicas, o algún otro grupo que quiera, de pronto, revisar algo?

**Andrés:** Yo yo tengo un par de preguntas todavía, como dentro de los retos que hemos encontrado, y es, nosotros hemos estado haciendo predicciones sobre tres municipios, ¿no? Y digamos, para decir de escoger un modelo uno sobre el otro, todavía no hemos como del todo definido como como cuáles realmente el valor total de error que se va a escoger. Porque, pues, digamos, para cada municipio tenemos el MAE o el RMC, bueno, sabemos que un modelo está creciendo mejor, no sé, para Ibagué. Pero para para ya poder comparar entre modelos, pues tenemos los tres municipios y no no hemos todavía visto con claridad si es bueno tomar el promedio de ese de esos errores. Flat, sin ponderación, o si se pondera por población, nos queda tan claro, sobre todo porque, por ejemplo, ponderar por población un incurriría ahí como en tal vez un tema ético en el que uno prioriza simplemente, una una ciudad o un municipio con más población a que prediga mejor ahí. Cuando, de pronto, un municipio muy pequeño precisamente, por ser pequeño, no tiene un sistema de salud superrobusto es al que más deberíamos utilizar tener como precisión ahí. Entonces, ahí no sé cómo usted lo lo veía.

**Carlos:** Ustedes están trabajando los tres lugares, comunidades, en vez ¿verdad? Por cada una ajustan un modelo distinto, ¿cierto?

**Andrés:** No, es es el mismo modelo, pero entonces, el...

**Carlos:** El modelo te arroja los tres los tres errores.

**Andrés:** Claro, pero, digamos, o sea, ustedes cogen y corren el modelo con Ibagué, y luego corren el modelo con, ajustan el voice. Y luego, independientemente, lo ajustan con el espinal. O sea, el modelo de baguette del ITC, y el modelo del espinal del IGBM tienen distintos pesos y distintos parámetros. ¿Cierto? Digamos que que, o sea, sí, sí tienes distinto o sea, distinto, dependiendo cómo es lo que se quiere inferir, sí, tienen distinto comportamiento los modelos. Y van a arrojar distintos RMCs.

**Carlos:** Y lo pensamos como una regresión lineal, yo cojo y cojo los datos de Ibagué y estimo la pendiente del interceptor para Ibagué con los datos de Ibagué, y luego cojo el final y estimo los datos de los parámetros de la pendiente de el intercepto para los datos del espinal, ¿cierto?

**Andrés:** Sí.

**Carlos:** O sea, son independientes los ajustes de los modelos en cada Perfecto. Y les están dando que en algún es mejor modelo que otro?

**Andrés:** Sí, el, o sea, se suele tener menores niveles de de de error, perdón, en en Espinal O En Girardot, y Vague suele tener errores bastante altos.

**Carlos:** Claro, claro. Sí, porque las dinámicas de Ibagué pueden ser un poco más complejas, también tengan cuenta que Ibagué tiene un gradiente altitudinal enorme. ¿No? Y tiene eso en la parte baja, que es está full valle de mandarinas, está parte alta, que ya es más parte andina. Eso, pues, esa heterogeneidad es ahí. Pero ¿cuál es mi recomendación ya final, Andrés? No los mezcle. Siga trabajando cada municipio, comunidad independiente, y calculan el RMC, calculan todas sus métricas por cada municipio. Y por cada municipio eligen el mejor modelo. ¿Listo? Pero sígannos manejando independientes, o sea, no hay necesidad de que ustedes, digamos, o sea, según le entiendo, digamos que ustedes tienen regresión de Poisson y la Y hacen regresión de Poisson y la luego para el espinal y luego para Girardón, digamos, ¿cierto? Y obtienen por cada modelo, en cada uno de los municipios, sus valores de sus métricas. Y lo que ustedes quieren ver es, bueno, para elegir cuál es el mejor modelo entre el o la regresión de Poisson Ah, ya. Medio de los tres, para ver en promedio cuál sirve mejor. Realmente, como ustedes están trabajando de manera independiente cada municipio, piénselo como si a cada municipio a usted le fuera a dar una herramienta distinta. Trabájelo como una tabla que tenga los tres municipios y reporte su propio sus distintas métricas por cada municipio. Hicieron municipios sin es mejor la LGBTI, y si en el espizal es mejor la regresión de Poisson,

**Jhon:** Si.

**Carlos:** está bien.

**Andrés:** Ah, no, ya ya ya está entendiendo un poco el punto suyo pero no, entonces, creo que de pronto nosotros no lo estamos haciendo como como usted dice, sino cogemos un mismo modelo con mismos hiperparámetros los tres corremos digamos, hacemos los entrenamientos, se se mide con la diferencia de los tres municipios, ¿sí? Tres municipios.

**Carlos:** Digamos que yo, cuando hago el entrenamiento, meten los datos de los tres municipios.

**Andrés:** Pues nosotros entrenamos con los datos de todos los municipios, pero inferimos sobre esos tres, porque, pues, decidimos que, entre más datos, mejor. Y, pues, por ahí los modelos capturaban como de todos los municipios, cuál es... Pero, en esencia, digamos que a a la pregunta que va es si la inferencia se hace sobre los tres municipios, con los con un mismo modelo mismos hiperparámetros. No, pues, digamos que no estamos cogiendo no sé, un TFT para ok, y luego cogemos un LiveJV en que funcionó mejor Espinal, y no, no estamos haciendo eso.

**Carlos:** Las métricas, Andrés, esa sería la idea, esa sería lo mejor, porque cogen cada municipio como una independiente y bien, pero se ya avanzaron con este otro pipeline y está bien. ¿Sí? No quiero como que se echen atrás ahorita, sino que si de pronto después se me y les queda tiempo, pues ahí sí miren a ver qué trae los by, pero ya tenían todo un entregable, ¿listo? Entonces, en ese si van a trabajar con esos errores promedios, lo mejor sería, incluso, en vez de promediarlos nomás. Realmente sería incluso hasta mejor sumarlos. Calcular lo colectivo para los tres municipios. Ya, sí, ok. Ok, listo. Yo hago con un error respecto porque usted acá están trabajando los tres municipios como si fuera una sola unidad. Porque no son independientes, están haciendo la diferencia en los tres ¿Listo? Ok.

**Andrés:** Listo. Listo. No, nos llevamos esa idea de de sumarlo, simplemente listo. Ok. Creo que no sé si hay alguien más de mi equipo, si tiene alguna otra duda. Si no tienen más dudas ustedes, yo tendría una una una pregunta, y es

**Carlos:** Claro, María.

**Amaya:** Nosotros sí, hicimos como la exploración que habíamos hablado la la vez que nos reunimos, y hicimos como dos modelos, tratamos de ver cómo nivel nacional, tratando de explorar como como una serie de tiempo con todos los municipios, y escogiendo como tres departamentos, pues, que para nosotros tenían una alta representatividad de los casos.

**Carlos:** Perfecto.

**Amaya:** Entonces, es pues nos han dado como como, pues los resultados son muy grandes, y con lo que ustedes, pues con lo que la discusión que que se tenido en la clase, pues hemos visto también parte del del del posible racional. Con la diversidad de los municipios. No, pues ahorita lo que los escuchábamos, no teníamos no habíamos incluido como la altura de los municipios como para cruzarse cosas de variables, a pesar de que intentamos como mantener como esa serie de y jugar como con el ciclovía del mosquito, el período de incubación de de la enfermedad, para el inicio de presentación de síntomas. Los resultados de los modelos sí son son son diferentes, En esa combinación

**Carlos:** Pero ¿cómo así? Tamaya? ¿Entre los distintos municipios o...?

**Amaya:** No, no, no, o sea, porque nosotros escogimos fueron dos modelos como para comparar un modelo escogiendo tres departamentos y haciendo como un promedio de los tres departamentos, o el modelo nacional. Contemplando alrededor de los novecientos municipios.

**Carlos:** Ok.

**Amaya:** Pero, pues, los valores de, cuando nosotros vemos los cuando nos da por analizar lo que es el en en o sea, cuando vemos el recall, la precisión, el f m un score y el ROC de del modelo uno, es el total de los municipios, versus el del modelo dos, que es el de los tres departamentos. Sí es muy diferente los resultados.

**Carlos:** ¿Cuál es la mejora, María?

**Amaya:** El de los tres departamentos.

**Carlos:** Ok, listo, Ahí te estás enfrentando algo que es superincreíble en el y que es como un tema muy interesante, y es la escala espacial. ¿Listo? ¿A qué me refiero con la escala espacial? Cuando uno, digamos, analiza datos a una escala de todo un país, ¿cierto? Yo estoy sumando tantos, digamos, variaciones individuales de cada municipio en ese colectivo de todo el país, que muchas veces van a haber ciertas variables que me van a explicar eso, que no necesariamente van a ser las mismas variables que nos lo explican a una escala departamental. ¿Me explico? A escala, digamos, en todo el país, puede que los comportamientos humanos no sean tan importantes. Mientras cascala de empresas de juez departamental, sí, o viceversa, claro. Cuando uno trabaja en esas dos escalas tan diferentes, unas escala nacional agregando todos los municipios, o una escala ya más enfocada en donde y usa el mismo set de variables para ambas escalas, y uno se da cuenta de esas diferencias en el performance al momento de hacer, digamos, las evaluaciones a lo que te está diciendo es que a esas dos escalas, escala nacional grande, cincuenta y pegre millones habitantes, reducimos toda la diversidad de climas tenemos en un clima promedio nacional, ¿cierto? Reducimos toda la diversidad que tenemos cultural y demás en una sola parte, digamos, socioeconómica ahí, van a haber ciertas variables que van a tener un mayor efecto, que eso limitamos a una escala un poco más localizada, que ya sería la escala departamental. ¿Listo, Amaya? Por ejemplo, en el Amazonas, en el departamento del Amazonas, el clima a ser muy homogéneo. Mientras que en el departamento de pollo acá tenemos desde Páramo hasta Magdalena Medio hacia la parte de Otanche hacia la digamos, que uno se descuelga para llegar a Santander o a Callas. O también en Boyacá tenemos que, de Montellano, en toda la falta Santa María, en toda la parte de Pajarito y demás. Entonces, mira que, según esas escalas que estemos usando, van a cambiar Mi recomendación hay, Amalia, es cácense con uno, miren cuáles está dando mejor y definan eso como lo que van a trabajar para hacer su sistema y trabajen ya eso, ¿listo?

**Amaya:** Perfecto, profe, gracias.

**Carlos:** Listo. ¿Alguna otra pregunta, María?

**Amaya:** Yo creo que por ahora no, porque la tenemos que definir como grupo es cuál de los dos modelos vamos a seguir para para montarse la parte del despliegue y igual cualquier cosa le escribimos por correo.

**Carlos:** Por eso saben que superatento. Y no es por saludarnos a Mario, pero no había dicho escala nacional. Podría ser muy interesante chismosear ahí, ¿listo?

**Amaya:** Perfecto, profe, gracias.

**Carlos:** Bueno, chicos y chicas, ¿algún otro grupo que tenga alguna duda, algo que comentar? Es una vez si, pues, si quieres la tratar o Fernando. Listo. Entonces, yo creo que todo claro con
