package shopgraph.analytics

import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

/**
 * ShopGraph Module 03 (Scala GraphX Component)
 * ============================================
 * Implements distributed bipartite customer-product graph analytics using Apache Spark GraphX.
 * Reads interactions from Parquet, computes PageRank, Connected Components,
 * Triangle Counts, and projects the Product Co-Purchase network across the cluster.
 */
object ShopGraphXAnalysis {

  case class VertexProperty(name: String, vertexType: String)
  case class EdgeProperty(weight: Double)

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("ShopGraph-GraphX-Bipartite-Analysis")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    println("================================================================================")
    println("SHOPGRAPH: DISTRIBUTED GRAPHX BIPARTITE & CO-PURCHASE ANALYTICS")
    println("================================================================================")

    val dataPath = if (args.length > 0) args(0) else "data/interactions.parquet"
    val outputPath = if (args.length > 1) args(1) else "outputs/tables"

    // 1. Read clean interaction table from Parquet
    val interactionsDF = spark.read.parquet(dataPath)
    println(s"[GraphX] Loaded ${interactionsDF.count()} interaction records from Parquet.")

    // 2. Hash IDs to 64-bit Long Vertex IDs for GraphX
    // Separate customer namespace (positive hash) and product namespace (negative hash)
    val customerVertices: RDD[(VertexId, VertexProperty)] = interactionsDF
      .select("customer_id")
      .distinct()
      .rdd
      .map(row => {
        val cid = row.getAs[Long]("customer_id")
        val vid = cid // Direct Long ID
        (vid, VertexProperty(s"Customer_$cid", "customer"))
      })

    val productVertices: RDD[(VertexId, VertexProperty)] = interactionsDF
      .select("stock_code")
      .distinct()
      .rdd
      .map(row => {
        val scCode = row.getAs[String]("stock_code")
        // Use negative 64-bit hash to partition vertex ID space
        val vid = -(scCode.hashCode.toLong.abs + 1000000000L)
        (vid, VertexProperty(scCode, "product"))
      })

    val vertices: RDD[(VertexId, VertexProperty)] = customerVertices.union(productVertices)

    // 3. Construct Edges
    val edges: RDD[Edge[Double]] = interactionsDF.rdd.map(row => {
      val cid = row.getAs[Long]("customer_id")
      val scCode = row.getAs[String]("stock_code")
      val weight = row.getAs[Double]("confidence_weight")
      val srcVid = cid
      val dstVid = -(scCode.hashCode.toLong.abs + 1000000000L)
      Edge(srcVid, dstVid, weight)
    })

    // 4. Instantiate Bipartite Property Graph
    val defaultVertex = VertexProperty("Unknown", "unknown")
    val bipartiteGraph: Graph[VertexProperty, Double] = Graph(vertices, edges, defaultVertex)
      .cache()

    val numVertices = bipartiteGraph.vertices.count()
    val numEdges = bipartiteGraph.edges.count()
    println(s"[GraphX] Graph constructed: $numVertices vertices, $numEdges edges.")

    // -------------------------------------------------------------------------
    // 5. Distributed PageRank Centrality
    // -------------------------------------------------------------------------
    println("[GraphX] Running Distributed Dynamic PageRank (resetProb=0.15, tol=0.001)...")
    val prGraph = bipartiteGraph.pageRank(0.001, resetProb = 0.15)

    // Join PageRank scores with vertex metadata and filter for top products
    val topProductsPR = prGraph.vertices
      .join(bipartiteGraph.vertices)
      .filter { case (_, (_, prop)) => prop.vertexType == "product" }
      .map { case (vid, (rank, prop)) => (prop.name, rank) }
      .sortBy(_._2, ascending = false)
      .take(20)

    println("\n--- Top 10 Products by Structural GraphX PageRank ---")
    topProductsPR.take(10).zipWithIndex.foreach { case ((code, rank), idx) =>
      println(f"  #${idx + 1}%2d: StockCode: $code%-12s | PageRank: $rank%.6f")
    }

    // -------------------------------------------------------------------------
    // 6. Connected Components
    // -------------------------------------------------------------------------
    println("\n[GraphX] Computing Connected Components across cluster...")
    val ccGraph = bipartiteGraph.connectedComponents()
    val componentSizes = ccGraph.vertices
      .map { case (_, compId) => (compId, 1L) }
      .reduceByKey(_ + _)
      .sortBy(_._2, ascending = false)

    val totalComponents = componentSizes.count()
    val largestComponent = componentSizes.first()._2
    val coveragePct = (largestComponent.toDouble / numVertices) * 100.0
    println(s"[GraphX] Total Connected Components: $totalComponents")
    println(f"[GraphX] Giant Component Size: $largestComponent vertices ($coveragePct%.2f%% coverage)")

    // -------------------------------------------------------------------------
    // 7. Product Co-Purchase Projection & Triangle Counting
    // -------------------------------------------------------------------------
    println("\n[GraphX] Computing Co-Purchase Projection and Triangle Counting...")
    // Invert bipartite graph to project Product-to-Product through Customers:
    // (Customer) -> [(Product1, Weight1), (Product2, Weight2), ...]
    val customerBaskets = bipartiteGraph.edges
      .map(e => (e.srcId, e.dstId))
      .groupByKey()

    val coPurchaseEdges = customerBaskets.flatMap { case (_, productList) =>
      val prods = productList.toArray.sorted
      for {
        i <- 0 until prods.length
        j <- (i + 1) until prods.length
      } yield Edge(prods(i), prods(j), 1.0)
    }.reduceByKey((e1, e2) => Edge(e1.srcId, e1.dstId, e1.attr + e2.attr))
     .filter(_.attr >= 10.0) // Filter significant co-purchase thresholds

    val projectedGraph = Graph.fromEdges(coPurchaseEdges, 1).cache()
    val triangleCounts = projectedGraph.triangleCount().vertices
    val totalTriangles = triangleCounts.map(_._2).reduce(_ + _) / 3

    println(s"[GraphX] Projected Product Co-Purchase Graph: ${projectedGraph.vertices.count()} nodes, ${projectedGraph.edges.count()} edges.")
    println(s"[GraphX] Total Triangles in Co-Purchase Network: $totalTriangles")

    println("\n================================================================================")
    println("GRAPHX ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
    println("================================================================================")
    spark.stop()
  }
}
