ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"

lazy val root = (project in file("."))
  .settings(
    name := "CS441_Fall2024_Assignment",
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) =>
        xs match {
          case "MANIFEST.MF" :: Nil => MergeStrategy.discard
          case "services" :: _      => MergeStrategy.concat
          case _                    => MergeStrategy.discard
        }
      case "reference.conf" => MergeStrategy.concat
      case x if x.endsWith(".proto") => MergeStrategy.rename
      case x if x.contains("hadoop") => MergeStrategy.first
      case _ => MergeStrategy.first
    },

    assembly / assemblyOption := (assembly / assemblyOption).value.withIncludeScala(true),
//    javaOptions ++= Seq(
//      "-Djavacpp.platform=macosx-arm64",
//      s"-Djava.library.path=${System.getProperty("user.home")}/.ivy2/cache/org.bytedeco.javacpp-presets/openblas/jar/:"
//    ),
    Compile / run / fork := true,

    libraryDependencies ++= Seq(
      "com.knuddels" % "jtokkit" % "1.1.0",
      "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1",
      "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
      "org.nd4j" % "nd4j-native" % "1.0.0-M2.1",
      "org.apache.hadoop" % "hadoop-common" % "3.3.6",
      "org.apache.hadoop" % "hadoop-hdfs" % "3.3.6",
      "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.3.6",
      "org.apache.hadoop" % "hadoop-mapreduce-client-common" % "3.3.6",
      "org.apache.hadoop" % "hadoop-mapreduce-client-jobclient" % "3.3.6",
      "org.apache.hadoop" % "hadoop-yarn-common" % "3.3.6",
      "org.slf4j" % "slf4j-simple" % "2.0.13",
      "com.typesafe" % "config" % "1.4.3"
    ),
  )