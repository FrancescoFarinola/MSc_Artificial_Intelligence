import React from "react";
import Navbar from "../components/Navbar";
import CustomizedInputBase from "../components/Search";
import { useStyles } from "../components/Styles";
import Loader from "../components/Loader";
import "antd/dist/antd.css";
import { Table } from "antd";

const columns = [
  { title: "UPC", dataIndex: "upc1", key: "upc1" },
  { title: "Product ID", dataIndex: "productID", key: "productID" },
  { title: "Product Name", dataIndex: "productName", key: "productName" },
  { title: "Price", dataIndex: "productFinalPrice", key: "productFinalPrice"}
];

const sub_columns = [
  { title: "Farm Name", dataIndex: "farmName", key: "farmName" },
  { title: "Latitude", dataIndex: "farmLatitude", key: "farmLatitude" },
  { title: "Longitude", dataIndex: "farmLongitude", key: "farmLongitude" },
  { title: "Material Name", dataIndex: "materialName", key: "materialName"},
  { title: "Harvest Time", dataIndex: "harvestTime", key: "harvestTime"},
];

export default function Explorer(props) {
  const classes = useStyles();
  const supplyChainContract = props.supplyChainContract;
  const [productData, setProductData] = React.useState([]);
  const [productHistory, setProductHistory] = React.useState([]);
  const [Text, setText] = React.useState(false);
  const navItem = [];
  const [loading, setLoading] = React.useState(false);

  const findProduct = async (search) => {
    var arr = [];
    var temp = [];
    setLoading(true);
    try {
      setProductData([]);
      setProductHistory([]);
      var a = await supplyChainContract.methods
        .fetchProduct(parseInt(search)).call();
      temp.push(a);
      setProductData(temp);
      console.log(temp)
      arr = [];
      var l = await supplyChainContract.methods
        .fetchLengthMaterials(parseInt(search)).call();
      var ids = await supplyChainContract.methods
        .fetchProductLots(parseInt(search)).call();
      arr = [];
      for (var i = 0; i < l; i++) {
        var h = await supplyChainContract.methods
          .fetchMaterial(parseInt(ids[i])).call();
        arr.push(h);
      }
      console.log(arr)

      setProductHistory(arr);
    } catch (e) {
      setText(true);
      console.log(e);
    }
    setLoading(false);
  };

  return (
    <>
      <Navbar navItems={navItem}>
        {loading ? (
          <Loader />
        ) : (
          <>
            <h1 className={classes.pageHeading}>Search a product</h1>
            <CustomizedInputBase findProduct={findProduct} />
            {productData.length !== 0 ? (
              <>
                <Table
                  columns={columns}
                  expandedRowRender={record => (
                    <Table columns={sub_columns} dataSource={productHistory} pagination={true} />
                  )}
                  dataSource={productData}
                />
                    
              </>
            ) : (
              <>{Text ? <p>Product Not Found</p> : <></>}</>
            )}
          </>
        )}
      </Navbar>
    </>
  );
}