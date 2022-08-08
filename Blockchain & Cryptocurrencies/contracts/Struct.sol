// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

library Struct {
    enum MaterialState
    {
      Harvested,
      Packed,
      BoughtByManufacturer
    }

    enum ProductState
    {     
      Processed,           
      PackedByManufacturer,    
      BoughtByDistributor,       
      Shipped,               
      ReceivedByRetailer,      
      Placed,                
      Purchased    
    }

    struct FarmerDetails{
      address payable farmer;
      string  farmerName;
      string  farmerLatitude;
      string  farmerLongitude;
    }

    struct ManufactureDetails {
      address payable manufacturer;
      string manufacturerName;
    }

    struct DistributorDetails {
      address payable distributor;
    }

    struct RetailerDetails {
      address payable retailer;
    }

    struct ConsumerDetails {
      address payable consumer;
    }

    struct Material {
      address owner; 
      FarmerDetails farmer;
      ManufactureDetails manufacturer;
      uint    lotNumber;
      uint    materialID;
      string  materialName;
      uint    materialMaxQuantity;
      uint    materialQuantity;
      uint    materialPrice;
      uint    harvestTime;
      MaterialState mState;
    }

    struct Product {
      ManufactureDetails manufacturer;
      DistributorDetails distributor;
      RetailerDetails retailer;
      ConsumerDetails consumer;
      address owner;
      uint    upc;
      uint    sku;
      uint    productID;
      string  productName;
      uint    productBasePrice;
      uint    productFinalPrice;
      uint[]  materialsUsed;
      ProductState pState;
    }

    struct Roles {
      bool Farmer;
      bool Manufacturer;
      bool Distributor;
      bool Retailer;
      bool Consumer;
    }

}