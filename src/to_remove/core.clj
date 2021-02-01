(ns to-remove.core
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.xgboost]
            [tech.v3.ml :as ml]
            [clj-http.client :as http]
            [clojure.tools.logging :as log]))

(def ds (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv"))

(def numeric-ds (ds/categorical->number ds cf/categorical))

(def regression-ds (ds-mod/set-inference-target numeric-ds "petal_width"))

(def model (ml/train-split regression-ds {:model-type :xgboost/regression}))

(prn (:loss model))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
