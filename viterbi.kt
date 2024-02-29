import kotlin.math.*

fun gaussianProb(x: Double, paraTuple: Pair<Double?, Double?>): Double {
    if (paraTuple.toList() == listOf(null, null)) {
        return 0.0
    }

    val (mean, std) = paraTuple
    val gaussianPercentile = (2 * PI * (std!! + 0.01).pow(2)).pow(-0.5) * exp(-(x - mean!!).pow(2) / (2 * (std + 0.01).pow(2)))
    return gaussianPercentile
}

fun multidimensionalViterbi(
    evidenceVector: List<List<Double>>,
    states: List<String>,
    priorProbs: Map<String, Double>,
    transitionProbs: Map<String, Map<String, List<Double>>>,
    emissionParas: Map<String, List<Double>>,
    numDimensions: Int = 2
): Pair<List<String>, Double> {
    val sequence = mutableListOf<String>()
    var probability = 0.0

    if (evidenceVector.isEmpty()) {
        return sequence to probability
    }

    val nl = mutableListOf<List<Double>>()

    for (i in states.indices) {
        val priorProb = priorProbs[states[i]] ?: 0.0
        val priorProbLog = if (priorProb != 0.0) ln(priorProb) else Double.NEGATIVE_INFINITY
        val gaussianProbAll = mutableListOf<Double>()
        for (z in 0 until numDimensions) {
            gaussianProbAll.add(gaussianProb(evidenceVector[0][z], emissionParas[states[i]]!!))
            gaussianProbAll[z] = if (gaussianProbAll[z] != 0.0) ln(gaussianProbAll[z]) else Double.NEGATIVE_INFINITY
        }
        var gaussianProbSum = 0.0
        for (z in 0 until numDimensions) {
            gaussianProbSum += gaussianProbAll[z]
        }
        nl.add(listOf(priorProbLog + gaussianProbSum) + List(evidenceVector.size - 1) { 0.0 })
    }

    for (i in 1 until evidenceVector.size) {
        for (j in states.indices) {
            val state = states[j]
            val (maxVal, bestPrevProb, kNew) = if (j >= 1 && i >= 1) {
                var maxVal = Double.NEGATIVE_INFINITY
                var bestPrevProb: Double? = null
                var kNew: Int? = null
                for (k in states.indices) {
                    var transitionProbSum = 0.0
                    try {
                        val transitionProbAll = mutableListOf<Double>()
                        for (z in 0 until numDimensions) {
                            transitionProbs[states[k]]?.get(states[j])?.get(z)?.let {
                                transitionProbAll.add(it)
                            }
                            transitionProbAll[z] = if (transitionProbAll[z] != 0.0) ln(transitionProbAll[z]) else Double.NEGATIVE_INFINITY
                            transitionProbSum += transitionProbAll[z]
                        }
                    } catch (e: Exception) {
                    }
                    if (states[j] in transitionProbs[states[k]]!! && (nl[k][i - 1] + transitionProbSum) >= maxVal) {
                        maxVal = nl[k][i - 1] + transitionProbSum
                        bestPrevProb = nl[k][i - 1]
                        kNew = k
                    }
                }
                Triple(maxVal, bestPrevProb, kNew)
            } else {
                Triple(nl[j][i - 1], null, null)
            }
            val (prevProb, prevState) = if (i >= 1) {
                bestPrevProb to states[kNew ?: j]
            } else {
                nl[j][i - 1] to states[j]
            }
            val gaussianAll = mutableListOf<Double>()
            var gaussianSum = 0.0
            for (z in 0 until numDimensions) {
                gaussianAll.add(gaussianProb(evidenceVector[i][z], emissionParas[state]!!))
                gaussianAll[z] = if (gaussianAll[z] != 0.0) ln(gaussianAll[z]) else Double.NEGATIVE_INFINITY
                gaussianSum += gaussianAll[z]
            }
            val transitionAll = mutableListOf<Double>()
            var transitionSum = 0.0
            for (z in 0 until numDimensions) {
                transitionProbs[prevState]?.get(state)?.get(z)?.let {
                    transitionAll.add(it)
                }
                transitionAll[z] = if (transitionAll[z] != 0.0) ln(transitionAll[z]) else Double.NEGATIVE_INFINITY
                transitionSum += transitionAll[z]
            }
            nl[j][i] = prevProb!! + gaussianSum + transitionSum
        }
    }
    val newS = mutableListOf<Double>()
    val seq = mutableListOf<Pair<Int, Int>>()

    var highestProb = Double.NEGATIVE_INFINITY
    var highestProbIndex: Int? = null
    for (j in states.indices) {
        if (highestProb < nl[j].last()) {
            highestProb = nl[j].last()
            highestProbIndex = j
        }
    }
    newS.add(highestProb)
    sequence.add(states[highestProbIndex!!])
    probability = highestProb

    for (i in evidenceVector.size - 2 downTo 0) {
        var highestProb = Double.NEGATIVE_INFINITY
        var newHighestProb = Double.NEGATIVE_INFINITY
        var bestState: String? = null
        var nj: Int? = null
        var ni: Int? = null

        for (j in states.indices) {
            if (sequence[0] !in transitionProbs[states[j]]!!) {
                continue
            }
            val transitionAll = mutableListOf<Double>()
            var transitionSum = 0.0
            for (z in 0 until numDimensions) {
                transitionProbs[states[j]]?.get(sequence[0])?.get(1)?.let {
                    transitionAll.add(it)
                }
                transitionAll[z] = if (transitionAll[z] != 0.0) ln(transitionAll[z]) else Double.NEGATIVE_INFINITY
                transitionSum += transitionAll[z]
            }
            if ((nl[j][i] + transitionSum) > highestProb) {
                highestProb = nl[j][i] + transitionSum
                newHighestProb = nl[j][i]
                bestState = states[j]
                nj = j
                ni = i
            }
        }
        if (bestState != null) {
            sequence.add(0, bestState)
            newS.add(0, newHighestProb)
            seq.add(0, nj to ni)
        }
    }

    if (probability == 0.0) {
        return null to 0.0
    }

    return sequence to exp(probability)
}
