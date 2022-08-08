// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import './Struct.sol';

contract SupplyChain {
    uint public upc;
    uint sku;
    uint mID;
    address payable contractOwner;

    mapping (uint => Struct.Material) materials;
    mapping (uint => Struct.Product) products;
    mapping(address => Struct.Roles) roles;

    constructor () {
        contractOwner = payable(msg.sender);
        upc = 1;
        sku = 1;
        mID = 1;
    }

    //<----------------------       MANAGING ROLES      ---------------------->

    //Modifier for owner only functions
    modifier onlyOwner() {
        require(msg.sender == contractOwner, "Only the owner can perform this operation");
        _;
    }

    //Farmer
    event FarmerAdded(address indexed _account);
    event FarmerRemoved(address indexed _account);

    function hasFarmerRole(address _account) public view returns (bool) {
        require(_account != address(0));
        return roles[_account].Farmer;
    }

    function addFarmerRole(address _account) public onlyOwner {
        require(_account != address(0), "Address cannot be zero address");
        require(!hasFarmerRole(_account));
        roles[_account].Farmer = true;
        emit FarmerAdded(_account);
    }

    function revokeFarmerRole(address _account) public onlyOwner {
        require(_account != address(0), "Address cannot be zero address");
        require(hasFarmerRole(_account), "This address does not have this role yet");
        roles[_account].Farmer = false;
        emit FarmerRemoved(_account);
    }

    //Manufacturer
    event ManufacturerAdded(address indexed _account);
    event ManufacturerRemoved(address indexed _account);

    function hasManufacturerRole(address _account) public view returns (bool) {
        require(_account != address(0));
        return roles[_account].Manufacturer;
    }

    function addManufacturerRole(address _account) public onlyOwner {
        require(_account != address(0), "Address cannot be zero address");
        require(!hasManufacturerRole(_account));
        roles[_account].Manufacturer = true;
        emit ManufacturerAdded(_account);
    }

    function revokeManufacturerRole(address _account) public onlyOwner {
        require(_account != address(0), "Address cannot be zero address");
        require(hasManufacturerRole(_account), "This address does not have this role yet");
        roles[_account].Manufacturer = false;
        emit ManufacturerRemoved(_account);
    }

    //Distributor
    event DistributorAdded(address indexed _account);
    event DistributorRemoved(address indexed _account);

    function hasDistributorRole(address _account) public view returns (bool) {
        require(_account != address(0));
        return roles[_account].Distributor;
    }

    function addDistributorRole(address _account) public onlyOwner {
        require(_account != address(0), "Address cannot be zero address");
        require(!hasDistributorRole(_account));
        roles[_account].Distributor = true;
        emit DistributorAdded(_account);
    }

    function revokeDistributorRole(address _account) public onlyOwner {
        require(_account != address(0), "Address cannot be zero address");
        require(hasDistributorRole(_account), "This address does not have this role yet");
        roles[_account].Distributor = false;
        emit DistributorRemoved(_account);
    }

    //Retailer
    event RetailerAdded(address indexed _account);
    event RetailerRemoved(address indexed _account);

    function hasRetailerRole(address _account) public view returns (bool) {
        require(_account != address(0));
        return roles[_account].Retailer;
    }

    function addRetailerRole(address _account) public onlyOwner {
        require(_account != address(0), "Address cannot be zero address");
        require(!hasRetailerRole(_account));
        roles[_account].Retailer = true;
        emit RetailerAdded(_account);
    }

    function revokeRetailerRole(address _account) public onlyOwner {
        require(_account != address(0), "Address cannot be zero address");
        require(hasRetailerRole(_account), "This address does not have this role yet");
        roles[_account].Retailer = false;
        emit RetailerRemoved(_account);
    }
  
    //Consumer
    event ConsumerAdded(address indexed _account);
    event ConsumerRemoved(address indexed _account);

    function hasConsumerRole(address _account) public view returns (bool) {
        require(_account != address(0));
        return roles[_account].Consumer;
    }

    function addConsumerRole(address _account) public onlyOwner {
        require(_account != address(0), "Address cannot be zero address");
        require(!hasConsumerRole(_account));
        roles[_account].Consumer = true;
        emit ConsumerAdded(_account);
    }

    function revokeConsumerRole(address _account) public onlyOwner {
        require(_account != address(0), "Address cannot be zero address");
        require(hasConsumerRole(_account), "This address does not have this role yet");
        roles[_account].Consumer = false;
        emit ConsumerRemoved(_account);
    }

    //<----------------------       DEFINING EVENTS      ---------------------->

    event Harvested(uint lotNumber);
    event Packed(uint lotNumber);
    event BoughtByManufacturer(uint lotNumber);
    event Processed(uint upc);
    event PackedByManufacturer(uint upc);
    event BoughtByDistributor(uint upc);
    event Shipped(uint upc);
    event ReceivedByRetailer(uint upc);
    event Placed(uint upc);
    event Purchased(uint upc);

    //<----------------------       FUNCTION MODIFIERS      ---------------------->

    // Define a modifer that verifies the Caller
    modifier verifyCaller (address _address) {
        require(msg.sender == _address, "This account is not the owner of this item");
        _;
    }

    // Define a modifier that checks if the paid amount is sufficient to cover the price
    modifier paidEnough(uint _price) {
        require(msg.value >= _price, "The amount sent is not sufficient for the price");
        _;
    }

    modifier verifyQuantity(uint _lotNumber, uint _quantity) {
        require(materials[_lotNumber].materialMaxQuantity >= _quantity, "The quantity required is not enough");
        _;
        uint maxQuantity = materials[_lotNumber].materialMaxQuantity;
        materials[_lotNumber].materialMaxQuantity = maxQuantity - _quantity;
    }

    // Define a modifier that checks the price and refunds the remaining balance
    modifier checkValueForManufacturer(uint _lotNumber) {
        _;
        uint price = materials[_lotNumber].materialPrice;
        uint amountToReturn = msg.value - price;
        materials[_lotNumber].manufacturer.manufacturer.transfer(amountToReturn);
    }

    // Define a modifier that checks the price and refunds the remaining balance
    modifier checkValueForDistributor(uint _upc) {
        _;
        uint price = products[_upc].productBasePrice;
        uint amountToReturn = msg.value - price;
        products[_upc].distributor.distributor.transfer(amountToReturn);
    }

    // Define a modifier that checks the price and refunds the remaining balance
    // to the Consumer
    modifier checkValueForConsumer(uint _upc) {
        _;
        uint _price = products[_upc].productFinalPrice;
        uint amountToReturn = msg.value - _price;
        products[_upc].consumer.consumer.transfer(amountToReturn);
    }

    modifier checkMaterialsOwner(uint[] memory _lotNumbers) {
        for(uint i=0; i < _lotNumbers.length; i++) {
        uint rmID = _lotNumbers[i];
        require(materials[rmID].owner == msg.sender, "The Manufacturer does not own the current Material!");
        }
        _;
    }

    modifier harvested(uint _lotNumber) {
        require(materials[_lotNumber].mState == Struct.MaterialState.Harvested, "The Material is not in Harvested state!");
        _;
    }

    modifier packed(uint _lotNumber) {
        require(materials[_lotNumber].mState == Struct.MaterialState.Packed, "The Material is not in Packed state!");
        _;
    }

    modifier processed(uint _upc) {
        require(products[_upc].pState == Struct.ProductState.Processed, "The Product is not yet in Processed state!");
        _;
    }

    modifier packed_by_manufacturer(uint _upc) {
        require(products[_upc].pState == Struct.ProductState.PackedByManufacturer, "The Product is not in PackedByManufacturer state!");
        _;
    }

    modifier bought_by_distributor(uint _upc) {
        require(products[_upc].pState == Struct.ProductState.BoughtByDistributor, "The Product is not in BoughtByDistributor state!");
        _;
    }

    modifier shipped(uint _upc) {
        require(products[_upc].pState == Struct.ProductState.Shipped, "The Product is not in Shipped state!");
        _;
    }

    modifier received_by_retailer(uint _upc) {
        require(products[_upc].pState == Struct.ProductState.ReceivedByRetailer, "The Product is not in ReceivedByRetailer state!");
        _;
    }

    modifier placed(uint _upc) {
        require(products[_upc].pState == Struct.ProductState.Placed, "The Product is not in Placed state!");
        _;
    }

    modifier purchased(uint _upc) {
        require(products[_upc].pState == Struct.ProductState.Purchased, "The Product is not in Purchased state!");
        _;
    }

    //<----------------------       FUNCTIONS      ---------------------->

    function harvestMaterial(string  memory _farmName,  
                            string  memory _farmLatitude,
                            string  memory _farmLongitude,
                            string  memory _materialName,
                            uint    _maxQuantity) 
                            public 
                            returns (uint)
    {
        require(hasFarmerRole(msg.sender), 'You need to be a farmer');
        // Assign new Material fields
        Struct.Material memory newMaterial;
        newMaterial.owner = msg.sender;
        newMaterial.farmer.farmer = payable(msg.sender);
        newMaterial.farmer.farmerName = _farmName;
        newMaterial.farmer.farmerLatitude = _farmLatitude;
        newMaterial.farmer.farmerLongitude = _farmLongitude;
        newMaterial.materialName = _materialName;
        newMaterial.materialMaxQuantity = _maxQuantity;
        // Assign unique lot number based on progreessive number and timestamp
        newMaterial.materialID = mID;
        newMaterial.harvestTime = block.timestamp;
        uint rmID = mID + newMaterial.harvestTime;
        newMaterial.lotNumber = rmID;
        // Increase progressive number 
        mID = mID + 1;
        // Setting state
        newMaterial.mState = Struct.MaterialState.Harvested;
        // Adding new Item to map
        materials[rmID] = newMaterial;
        // Emit the appropriate event
        emit Harvested(rmID);
        return (rmID);
    }

    function packMaterial(uint _lotNumber,
                            uint _quantity,
                            uint _price)
                            public 
                            harvested(_lotNumber) 
                            verifyCaller(materials[_lotNumber].owner) 
                            verifyQuantity(_lotNumber, _quantity)
    {
        require(hasFarmerRole(msg.sender), 'You need to be a farmer');
        // Update the appropriate fields
        Struct.Material storage existingMaterial = materials[_lotNumber];
        existingMaterial.materialQuantity = _quantity;
        existingMaterial.materialPrice = _price;
        existingMaterial.mState = Struct.MaterialState.Packed;
        // Emit the appropriate event
        emit Packed(_lotNumber);
    }

    // Use the above defined modifiers to check if the item is available for sale, if the buyer has paid enough,
    // and any excess ether sent is refunded back to the buyer
    function buyMaterial(uint _lotNumber) 
                        public 
                        payable  
                        packed(_lotNumber) 
                        paidEnough(materials[_lotNumber].materialPrice) 
                        checkValueForManufacturer(_lotNumber)
        {
        require(hasManufacturerRole(msg.sender), 'You need to be a manufacturer');
        // Update the appropriate fields - ownerID, distributorID, itemState
        Struct.Material storage existingMaterial = materials[_lotNumber];
        existingMaterial.owner = msg.sender;
        existingMaterial.manufacturer.manufacturer = payable(msg.sender);
        existingMaterial.mState = Struct.MaterialState.BoughtByManufacturer;
        // Transfer money to farmer
        uint price = materials[_lotNumber].materialPrice;
        materials[_lotNumber].farmer.farmer.transfer(price);
        // emit the appropriate event
        emit BoughtByManufacturer(_lotNumber);
    }


    function processProduct(uint    _upc,
                            uint[]  memory _lotNumbers,
                            string  memory _productName)
                            public
                            checkMaterialsOwner(_lotNumbers)
    {
        require(hasManufacturerRole(msg.sender), 'You need to be a manufacturer');
        Struct.Product memory newProduct;
        newProduct.owner = msg.sender;
        newProduct.manufacturer.manufacturer = payable(msg.sender);
        newProduct.upc = _upc;
        newProduct.sku = sku;
        newProduct.productID = _upc + sku;
        newProduct.materialsUsed = _lotNumbers;
        newProduct.productName = _productName;
        newProduct.pState = Struct.ProductState.Processed;
        // Increment sku
        sku = sku + 1;
        // Add newProduct to mapping
        products[_upc] = newProduct;
        // Emit proper event
        emit Processed(_upc);
    }


    function packProduct(uint _upc,
                        uint _productBasePrice)
                        public
                        processed(_upc)
                        verifyCaller(products[_upc].owner) 
    {
        require(hasManufacturerRole(msg.sender), 'You need to be a manufacturer');
        Struct.Product storage existingProduct = products[_upc];
        existingProduct.productBasePrice = _productBasePrice;
        existingProduct.pState = Struct.ProductState.PackedByManufacturer;
        emit PackedByManufacturer(_upc);
    }


    function buyProduct(uint _upc) 
                        public
                        payable
                        packed_by_manufacturer(_upc)
                        paidEnough(products[_upc].productBasePrice) 
                        checkValueForDistributor(_upc)
    {
        require(hasDistributorRole(msg.sender), 'You need to be a distributor');
        // Update the appropriate fields
        Struct.Product storage existingProduct = products[_upc];
        existingProduct.owner = msg.sender;
        existingProduct.distributor.distributor = payable(msg.sender);
        existingProduct.pState = Struct.ProductState.BoughtByDistributor;
        // Transfer money to farmer
        uint productPrice = products[_upc].productBasePrice;
        products[_upc].manufacturer.manufacturer.transfer(productPrice);
        // Emit the appropriate event
        emit BoughtByDistributor(_upc);
    }


    function shipProduct(uint _upc) 
                        public
                        bought_by_distributor(_upc)
                        verifyCaller(products[_upc].owner)
    {
        require(hasDistributorRole(msg.sender), 'You need to be a distributor');
        // Update the appropriate fields
        Struct.Product storage existingProduct = products[_upc];
        existingProduct.pState = Struct.ProductState.Shipped;
        // Emit the appropriate event
        emit Shipped(_upc);
    }


    function receiveProduct(uint _upc) 
                            public
                            shipped(_upc)
    {
        require(hasRetailerRole(msg.sender), 'You need to be a retailer');
        // Update the appropriate fields - ownerID, retailerID, itemState
        Struct.Product storage existingProduct = products[_upc];
        existingProduct.owner = msg.sender;
        existingProduct.retailer.retailer = payable(msg.sender);
        existingProduct.pState = Struct.ProductState.ReceivedByRetailer;
        // Emit the appropriate event
        emit ReceivedByRetailer(_upc);
    }


    function placeProduct(uint _upc,
                          uint _productFinalPrice)
                          public
                          received_by_retailer(_upc)
                          verifyCaller(products[_upc].owner)
    {
        require(hasRetailerRole(msg.sender), 'You need to be a retailer');
        // Update the appropriate fields - ownerID, retailerID, itemState
        Struct.Product storage existingProduct = products[_upc];
        existingProduct.productFinalPrice = _productFinalPrice;
        existingProduct.pState = Struct.ProductState.Placed;
        // Emit the appropriate event
        emit Placed(_upc);
    }

    // Define a function 'purchaseItem' that allows the consumer to mark an item 'Purchased'
    // Use the above modifiers to check if the item is received
    function purchaseProduct(uint _upc) 
                            public 
                            payable 
                            placed(_upc)
                            paidEnough(products[_upc].productFinalPrice)
                            checkValueForConsumer(_upc)
    {
        require(hasConsumerRole(msg.sender), 'You need to be a consumer');
        // Update the appropriate fields - ownerID, consumerID, itemState
        Struct.Product storage existingProduct = products[_upc];
        existingProduct.owner = msg.sender;
        existingProduct.consumer.consumer = payable(msg.sender);
        existingProduct.pState = Struct.ProductState.Purchased;
        // Emit the appropriate event
        emit Purchased(_upc);
    }

    // Define a function 'fetchItemBufferTwo' that fetches the data
    function fetchMaterial(uint _lotNumber) public view returns 
    (
    string memory farmName,
    string memory farmLatitude,
    string memory farmLongitude,
    string memory materialName,
    uint    harvestTime
    ) 
    {
        // Assign values to the 9 parameters
    Struct.Material memory existingMaterial = materials[_lotNumber];
    farmName = existingMaterial.farmer.farmerName;
    farmLatitude = existingMaterial.farmer.farmerLatitude;
    farmLongitude = existingMaterial.farmer.farmerLongitude;
    materialName = existingMaterial.materialName;
    harvestTime = existingMaterial.harvestTime;
    
    return 
    (
    farmName,
    farmLatitude,
    farmLongitude,
    materialName,
    harvestTime
    );
    }

    function fetchLengthMaterials(uint _upc) public view returns
    (
        uint n
    )
    {
        return products[_upc].materialsUsed.length;
    }


    function fetchProductLots(uint _upc) public view returns 
    (
        uint[] memory lotNumbers
    )
    {
        Struct.Product memory existingProduct = products[_upc];
        lotNumbers = existingProduct.materialsUsed;
        return (lotNumbers);
    }

    function fetchProduct(uint _upc) public view returns
    (
        uint upc1,
        uint productID,
        string memory productName,
        uint productFinalPrice
    )
    {
        Struct.Product memory existingProduct = products[_upc];
        return (
            existingProduct.upc,
            existingProduct.productID,
            existingProduct.productName,
            existingProduct.productFinalPrice
        );
    }

}